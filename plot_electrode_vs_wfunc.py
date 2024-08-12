import argparse
#import pickle
import os
import sys
import pathlib
from typing import Optional, Sequence, Callable
#from dataclasses import dataclass, field

import ase
import pandas as pd
#from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from md_interface_sim import folder_exist, build_pd
#from md_interface_sim.run_nvt import plot_work_functions

#import numpy as np

#import ase.db as db
#from ase import Atoms
#from ase.md.nvtberendsen import NVTBerendsen
#from ase import units
#from ase.parallel import world, barrier, parprint

#from sqlite3 import OperationalError

#from gpaw import GPAW#, FermiDirac, PoissonSolver, Mixer
#from gpaw.utilities import h2gpts

import plotly.graph_objects as go
import plotly.express as px


def generalised_hydrogen_electrode(E: float, E_ref: float, n_proton: float, proton_pot: float, n_cat: float, cat_pot: float, work_func: float, pH: float, T: float): return E - E_ref - n_cat*cat_pot - 0.5*n_proton*proton_pot - n_proton*(work_func - 2.303 * (8.617*10**-5)*T*pH)


def make_trace(name, db: pd.DataFrame, ghe_lambda: Callable[[pd.Series], float]):
    go.Scatter(
        name=name,
        x=db['work_top'],
        y=db.apply(ghe_lambda, axis=0),
    )

def get_H_count(atoms: ase.Atoms,) -> int:
    return atoms.get_chemical_formula(mode='all').count('H') - 64


def main(dbs_dirs: Sequence[str], save_name, sim_names: Optional[Sequence[str]]=None, ph: float = 6):

    dbs_dirs, dbs_selection = list(zip(*(db_dir.split('@') if '@' in db_dir else [db_dir, None] for db_dir in dbs_dirs)))
    for db_dir in dbs_dirs:
        if not os.path.basename(db_dir) in os.listdir(db_path if len(db_path := os.path.dirname(db_dir)) > 0 else '.'): raise FileNotFoundError("Can't find database")

    dat_pd = {name: build_pd(db_dir, select_key=sel_key) for name, db_dir, sel_key in zip(sim_names if sim_names is not None else dbs_dirs, dbs_dirs, dbs_selection)}

    ghe = lambda pd_series: generalised_hydrogen_electrode(
        E=pd_series['energy'],
        E_ref=dat_pd[sim_names[0] if sim_names is not None else dbs_dirs[0]]['energy'].mean(),
        n_proton=get_H_count(pd_series.get('atoms')),
        proton_pot= -6.616893,
        n_cat=0, # gotta make a function for counting
        cat_pot=0,
        work_func=pd_series['work_top'],
        pH=ph,
        T=pd_series['temperature']
    )

    fig = go.Figure()
    for key, val in dat_pd.items():
        fig.add_trace(make_trace(
            name=key,
            db=val,
            ghe_lambda=ghe

        ))

    fig.update_layout(
#        showlegend=False,
        xaxis_title=r'$\Phi$',
        yaxis_title=r'$E_{GCHE}$',
    )

    folder_exist('plots')
    fig.write_html('plots/' + save_name + '.html', include_mathjax='cdn')


if __name__ == '__main__':
    parser =argparse.ArgumentParser()
    parser.add_argument('db', type=str,  nargs='+', help='selection, can be added after @, note that the format have to follow the ase db select method. first database is reference!!!')
    parser.add_argument('-names', '--sim_names', type=str, nargs='+', help='list of names to use for every database in the plot.')
    parser.add_argument('-ph', '--pH', type=float, default=6)
    parser.add_argument('save_name')
    args = parser.parse_args()

    main(dbs_dirs=args.db,
         save_name=args.save_name,
         sim_names=args.sim_names,
         ph=args.pH)
    