#partition=katla
#nprocshared=32
#mem=2000MB
#constrain='[v4|v5]'
import argparse
import pickle
import os
import sys
import pathlib
from typing import Optional
from dataclasses import dataclass, field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from md_interface_sim import folder_exist, build_pd
from md_interface_sim.run_nvt import plot_work_functions, db_observer

import numpy as np
import pandas as pd

import ase.db as db
from ase import Atoms
#from ase.md.nvtberendsen import NVTBerendsen
#from ase import units
#from ase.parallel import world, barrier, broadcast

#from sqlite3 import OperationalError

from gpaw import GPAW, FermiDirac, PoissonSolver, Mixer
from gpaw.utilities import h2gpts

#import plotly.graph_objects as go
#import plotly.express as px
#import plotly.graph_objects as go


def single_point(db_dir: str, row_index: int, mode: str, xc: str, kpts: tuple[int, int, int] = (1, 1, 1), from_amanda: bool = False):
    if not os.path.basename(db_dir) in os.listdir(db_path if len(db_path := os.path.dirname(db_dir)) > 0 else '.'): raise FileNotFoundError("Can't find database")
    if from_amanda:
        with db.connect(db_dir) as db_obj:
            row = db_obj.get(selection=f'id={row_index}')
            atoms: Atoms = row.toatoms()
            if 'cur_time' not in globals(): global cur_time
            globals()['cur_time'] = row.get('Step')
            temperature = row.get('Temperature')
            brendsen_tau = 200 #row.get('brendsen_tau')

    else:
        with db.connect(db_dir) as db_obj:
            row = db_obj.get(selection=f'id={row_index}')
            atoms: Atoms = row.toatoms()
            if 'cur_time' not in globals(): global cur_time
            globals()['cur_time'] = row.get('time')
            temperature = row.get('temperature')
            brendsen_tau = row.get('brendsen_tau')

    name = f'{atoms.symbols}_recalculate_{xc}_{mode}' + (f'_k{"-".join(map(str, kpts))}' if mode == 'pw' else '')

    kpts_dict = dict(kpts=kpts) if mode == 'pw' else dict()

    calc_par_dict = dict(
        mode=mode,
        basis='dzp',
        # setups={'Pt': '10'},
        xc=xc,
        #kpts=kpts,
        occupations=FermiDirac(0.1),
        poissonsolver={'dipolelayer': 'xy'},
        mixer=Mixer(beta=0.025, nmaxold=5, weight=50.0),
        gpts=h2gpts(0.18, atoms.cell, idiv=8),
        #    convergence={'energy': 2.0e-7, 'density': 1e-5}, # HIGH
        convergence={'energy': 5.0e-6, 'density': 8e-5},  # LOW
        #    parallel = dict(sl_auto = True), #Using Scalapack = 4x speedup!
        txt=f'{name}.txt',
        **kpts_dict
    )
    calc_pickle = str(pickle.dumps(calc_par_dict))

    atoms.set_calculator(GPAW(**calc_par_dict))
    atoms.get_potential_energy()
    #mean_elec_pot_z = broadcast(atoms.calc.get_electrostatic_potential().mean(1).mean(0) - fermi_E)

    new_db_dir = f'{name}.db'
    db_observer(atoms, new_db_dir, temperature, brendsen_tau, 0, calc_pickle)
    del atoms,


def main(db_dir: str, nr_segments: int, start_from: int, mode: str, xc: str, kpts: tuple[int, int, int] = (1, 1, 1), from_amanda: bool = False):
    if not os.path.basename(db_dir) in os.listdir(db_path if len(db_path := os.path.dirname(db_dir)) > 0 else '.'): raise FileNotFoundError("Can't find database")

    md_panda: pd.DataFrame = build_pd(db_dir)

    percentiles: pd.DataFrame = md_panda['wftop' if from_anmanda else 'work_top'].quantile(np.linspace(1, 99, nr_segments), interpolation='nearest')
    percen_list = percentiles['wftop' if from_anmanda else 'work_top'].to_list

    pd_cutout = pd.concat([md_panda.query(f'{"wftop" if from_anmanda else "work_top"} == @work_percen').head(1) for work_percen in percen_list])

    for row in pd_cutout.itertuples()[start_from:]:
        single_point(db_dir, getattr(row, 'id'),  mode, xc, kpts, from_amanda=from_amanda)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('db', type=str)
    parser.add_argument('nr_segments', type=int)
    parser.add_argument('-from', '--start_from', type=int, default=0)
    parser.add_argument('-XC', '--XC', type=str, default='RPBE')
    parser.add_argument('-k', '--kpts', nargs=3, default=(1, 1, 1), type=int)
    parser.add_argument('-m', '--mode', choices=('fd', 'lcao', 'pw'), default='lcao', type=str)
    parser.add_argument('--from_amanda', action='store_true')
    args = parser.parse_args()

    main(db_dir=args.db, nr_segments=args.nr_segments, start_from=args.start_from, xc=args.XC, kpts=args.kpts, mode=args.mode, from_amanda=args.from_amanda)