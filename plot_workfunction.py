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
from md_interface_sim import folder_exist
from md_interface_sim.run_nvt import plot_work_functions

import numpy as np

import ase.db as db
from ase import Atoms
from ase.md.nvtberendsen import NVTBerendsen
from ase import units
from ase.parallel import world, barrier, parprint

from sqlite3 import OperationalError

from gpaw import GPAW#, FermiDirac, PoissonSolver, Mixer
#from gpaw.utilities import h2gpts

import plotly.graph_objects as go
import plotly.express as px


def main(db_dir: str, index: int):
    if not os.path.basename(db_dir) in os.listdir(db_path if len(db_path := os.path.dirname(db_dir)) > 0 else '.'): raise FileNotFoundError("Can't find database")
    with db.connect(db_dir) as db_obj:
        row = db_obj.get(selection=f'id={index}')
        atoms: Atoms = row.toatoms()
        calc_pickle = eval(row.data.get('calc_pickle'))
        atoms.set_calculator(GPAW(**pickle.loads(calc_pickle)))

        #start_time = row.get('time')
        time_step = row.get('time_step_size')
        global cur_time
        cur_time = db_obj.get(selection=f'id={index}').get('time')

        free_atoms = len(atoms) - sum([len(con.index) for con in atoms.constraints]) # this will only work in newer version of ase and older fixatoms in older version, since indicies are called a in some places

        temperature = row.get('temperature')
        brendsen_tau = row.get('brendsen_tau')

    atoms.get_potential_energy()

    plot_work_functions(atoms, os.path.basename(db_dir).split('.')[0], cur_time)

    #fermi_E = atoms.calc.get_fermi_level()
    #el_pot = atoms.calc.get_electrostatic_potential().mean(1).mean(0)

    #parprint(f'the potential object have type: {type(el_pot)}')
    #parprint(f'the fermi lvl is {atoms.calc.get_fermi_level()}')
    #parprint()
    #parprint('printing work function twice')
    #parprint()
    #parprint(atoms.calc.get_electrostatic_potential().mean(1).mean(0))
    #parprint()
    #parprint(atoms.calc.get_electrostatic_potential().mean(1).mean(0))


if __name__ == '__main__':
    parser =argparse.ArgumentParser()
    parser.add_argument('db', type=str)
    parser.add_argument('row_index', type=int)
    args = parser.parse_args()

    main(db_dir=args.db, index=args.row_index)
