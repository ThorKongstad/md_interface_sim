#partition=katla
#nprocshared=32
#mem=2000MB
#constrain='[v1|v2|v3|v4|v5]'
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


@dataclass
class iteration_counter:
    i: int = field(default=0)

    def __add__(self, other):
        return self.i + other


@retry(retry=retry_if_exception_type(OperationalError), stop=stop_after_attempt(5), wait=wait_random(min=60, max=600)) # will retry randomly within 10 min
def db_observer(atoms: Atoms, database_dir: str, temperature: float, brendsen_tau:float, time_step_size: float, calc_par_pickle: Optional[bytes] = None,):
    mean_elec_pot_z = atoms.calc.get_electrostatic_potential().mean(1).mean(0)
    fermi_E = atoms.calc.get_fermi_level()
    work_function_top = mean_elec_pot_z[-5] - fermi_E # [-1 if not (atoms.calc.parameters.get('mode') if isinstance(atoms.calc.parameters.get('mode'), str) else atoms.calc.parameters.get('mode').get('name')).lower() == 'pw' else -3] - fermi_E
    work_function_bot = mean_elec_pot_z[4] - fermi_E #[0 if not (atoms.calc.parameters.get('mode') if isinstance(atoms.calc.parameters.get('mode'), str) else atoms.calc.parameters.get('mode').get('name')).lower() == 'pw' else 2] - fermi_E
    with db.connect(database_dir) as db_obj:
        global cur_time
        cur_time = db_obj.get(selection=f'id={len(db_obj)}').get('time') + time_step_size
        db_obj.write(atoms=atoms, kinitic_E=atoms.get_kinetic_energy(), fermi_E=fermi_E, work_top=work_function_top, work_bot=work_function_bot, temperature=temperature, brendsen_tau=brendsen_tau, time=cur_time, time_step_size=time_step_size, data=dict(calc_pickle=(str(calc_par_pickle) if calc_par_pickle else str(pickle.dumps(atoms.calc.parameters)))))


def plot_work_functions(atoms: Atoms, calculation_name: str, time_step: float):
    if world.rank == 0:
        fermi_E = atoms.get_calculator().get_fermi_level()

        mean_elec_pot_z = atoms.get_calculator().get_electrostatic_potential().mean(1).mean(0) - fermi_E
        z_axis = np.linspace(0, atoms.cell[2, 2], len(mean_elec_pot_z), endpoint=False)

        fig = px.line(
            x=z_axis,
            y=mean_elec_pot_z,
        )

        fig.update_layout(xaxis_title='Z', yaxis_title='work function')

        folder_exist(f'workfunc_plots_{calculation_name}')
        save_name = f'workfunc_plots_{calculation_name}/time_{time_step}'
        fig.write_html(save_name + '.html', include_mathjax='cdn')
    barrier()


def main(md_db: str, n_steps: int):
    if not os.path.basename(md_db) in os.listdir(db_path if len(db_path := os.path.dirname(md_db)) > 0 else '.'): raise FileNotFoundError("Can't find database")
    with db.connect(md_db) as db_obj:
        row = db_obj.get(selection=f'id={len(db_obj)}')
        atoms: Atoms = row.toatoms()
        calc_pickle = eval(row.data.get('calc_pickle'))
        atoms.set_calculator(GPAW(**pickle.loads(calc_pickle)))

        #start_time = row.get('time')
        time_step = row.get('time_step_size')
        cur_time = db_obj.get(selection=f'id={len(db_obj)}').get('time') + time_step

        free_atoms = len(atoms) - sum([len(con.index) for con in atoms.constraints]) # this will only work in newer version of ase and older fixatoms in older version, since indicies are called a in some places

        temperature = row.get('temperature')
        brendsen_tau = row.get('brendsen_tau')

        dyn = NVTBerendsen(atoms, time_step * units.fs, free_atoms * temperature / len(atoms), brendsen_tau * units.fs, True)
    dyn.attach(db_observer,
               atoms=atoms,
               database_dir=md_db,
               temperature=temperature,
               brendsen_tau=brendsen_tau,
               time_step_size=time_step,
               calc_par_pickle=calc_pickle,
               )

    dyn.attach(plot_work_functions,
               interval=500,
               atoms=atoms,
               calculation_name=os.path.basename(md_db).split('.')[0],
               time_step=cur_time
               )

    dyn.run(n_steps)


if __name__ =='__main__':
    parser =argparse.ArgumentParser()
    parser.add_argument('db_structure', type=str)
    parser.add_argument('calc_n_steps', type=int)
    args = parser.parse_args()


    main(args.db_structure, args.calc_n_steps)

