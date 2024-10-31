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

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))
from md_interface_sim import folder_exist

import numpy as np

import ase.db as db
from ase import Atoms
from ase.md.nvtberendsen import NVTBerendsen
from ase import units
from ase.parallel import world, barrier, broadcast

from sqlite3 import OperationalError

from gpaw import GPAW#, FermiDirac, PoissonSolver, Mixer
#from gpaw.utilities import h2gpts

import plotly.graph_objects as go
import plotly.express as px
import plotly.graph_objects as go



def plot_work_functions(atoms: Atoms, calculation_name: str, time_step: float):
#    if world.rank == 0:
    fermi_E = broadcast(atoms.calc.get_fermi_level())

    mean_elec_pot_z = broadcast(atoms.calc.get_electrostatic_potential().mean(1).mean(0) - fermi_E)
    #mean_elec_pot_z = tuple(map(lambda pot: pot - fermi_E,  atoms.calc.get_electrostatic_potential().mean(1).mean(0)))
    z_axis = np.linspace(0, atoms.cell[2, 2], len(mean_elec_pot_z), endpoint=False)

    #fig = px.line(
    #    x=z_axis,
    #    y=mean_elec_pot_z.astype(float),
    #)

    if world.rank == 0:
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            mode='lines',
            x=z_axis,
            y=mean_elec_pot_z,
            line=dict(
                color='black',
            ),
        ))

        fig.update_layout(xaxis_title='Z', yaxis_title='work function')

        folder_exist(f'workfunc_plots_{calculation_name}')
        save_name = f'workfunc_plots_{calculation_name}/time_{time_step}'
        fig.write_html(save_name + '.html', include_mathjax='cdn')
    barrier()


def main(md_db: str, cur_time: int, out_db: Optional[str] = None):
    if not os.path.basename(md_db) in os.listdir(db_path if len(db_path := os.path.dirname(md_db)) > 0 else '.'): raise FileNotFoundError("Can't find database")
    with db.connect(md_db) as db_obj:
        row = db_obj.get(selection=f'time={cur_time}')
        atoms: Atoms = row.toatoms()
        xc_calc_pickle = eval(row.data.get('xc_calc_pickle'))
        tb_calc_pickle = eval(row.data.get('tb_calc_pickle'))

        temperature = row.get('temperature')
        brendsen_tau = row.get('brendsen_tau')

    xc_calc_par: dict = pickle.loads(xc_calc_pickle)
    tb_calc_par: dict = pickle.loads(tb_calc_pickle)

    atoms.set_calculator(GPAW(**xc_calc_par))

    atoms.get_potential_energy()
    atoms.get_forces()

    mean_elec_pot_z = atoms.calc.get_electrostatic_potential().mean(1).mean(0)
    fermi_E = atoms.calc.get_fermi_level()
    work_function_top = mean_elec_pot_z[-5] - fermi_E
    work_function_bot = mean_elec_pot_z[4] - fermi_E

    if out_db is None:
        sti = os.path.dirname(md_db)
        name = os.path.basename(md_db).split('-' + tb_calc_par['method'])
        out_db = f'{sti}/{name}_{xc_calc_par["xc"]}_{xc_calc_par["mode"]}'+(f'_k{"-".join(map(str, xc_calc_par["kpts"]))}' if xc_calc_par["mode"] == 'pw' else '')

    with db.connect(out_db):
        db_obj.write(atoms=atoms, kinitic_E=atoms.get_kinetic_energy(), fermi_E=fermi_E, work_top=work_function_top, work_bot=work_function_bot, temperature=temperature, brendsen_tau=brendsen_tau, time=cur_time, data=dict(xc_calc_pickle=xc_calc_pickle))


if __name__ =='__main__':
    parser =argparse.ArgumentParser()
    parser.add_argument('db_structure', type=str)
    parser.add_argument('time', type=int)
    parser.add_argument('--db_out', type=str)

#    parser.add_argument('calc_n_steps', type=int)
#    parser.add_argument('--until', action='store_true', default=False)
    args = parser.parse_args()


    main(args.db_structure, args.calc_n_steps, args.until)

