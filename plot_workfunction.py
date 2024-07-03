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

import numpy as np

import ase.db as db
from ase import Atoms
from ase.md.nvtberendsen import NVTBerendsen
from ase import units
from ase.parallel import world, barrier

from sqlite3 import OperationalError

from gpaw import GPAW#, FermiDirac, PoissonSolver, Mixer
#from gpaw.utilities import h2gpts

import plotly.graph_objects as go
import plotly.express as px



def plot_work_functions(atoms: Atoms, calculation_name: str, time_step: float):
#    if world.rank == 0:
    fermi_E = atoms.get_calculator().get_fermi_level()

#    mean_elec_pot_z = atoms.get_calculator().get_electrostatic_potential().mean(1).mean(0) - fermi_E
    mean_elec_pot_z = tuple(map(lambda pot: pot - fermi_E,  atoms.get_calculator().get_electrostatic_potential().mean(1).mean(0)))
    z_axis = np.linspace(0, atoms.cell[2, 2], len(mean_elec_pot_z), endpoint=False)

    fig = px.line(
        x=z_axis,
        y=mean_elec_pot_z,
    )

    fig.update_layout(xaxis_title='Z', yaxis_title='work function')

    folder_exist(f'workfunc_plots_{calculation_name}')
    save_name = f'workfunc_plots_{calculation_name}/time_{time_step}'
    fig.write_html(save_name + '.html', include_mathjax='cdn')
#    barrier()


def main(db_dir: str, index: int):
    if not os.path.basename(db_dir) in os.listdir(db_path if len(db_path := os.path.dirname(db_dir)) > 0 else '.'): raise FileNotFoundError("Can't find database")
    with db.connect(db_dir) as db_obj:
        row = db_obj.get(selection=f'id={len(db_obj)}')
        atoms: Atoms = row.toatoms()
        calc_pickle = eval(row.data.get('calc_pickle'))
        atoms.set_calculator(GPAW(**pickle.loads(calc_pickle)))

        #start_time = row.get('time')
        time_step = row.get('time_step_size')
        global cur_time
        cur_time = db_obj.get(selection=f'id={len(db_obj)}').get('time') + time_step

        free_atoms = len(atoms) - sum([len(con.index) for con in atoms.constraints]) # this will only work in newer version of ase and older fixatoms in older version, since indicies are called a in some places

        temperature = row.get('temperature')
        brendsen_tau = row.get('brendsen_tau')

    el_pot = atoms.get_calculator().get_electrostatic_potential.mean(1).mean(0)

    print(f'the potential object have type: {type(el_pot)}')
    print(f'the fermi lvl is {atoms.get_calculator().get_fermi_level()}')
    print()
    print('printing work function twice')
    print()
    print(atoms.get_calculator().get_electrostatic_potential.mean(1).mean(0))
    print()
    print(atoms.get_calculator().get_electrostatic_potential.mean(1).mean(0))




if __name__ == '__main__':
    parser =argparse.ArgumentParser()
    parser.add_argument('db', type=str)
    parser.add_argument('row_index', type=int)
    args = parser.parse_args()

    main(db_dir=args.db, index=args.row_index)
