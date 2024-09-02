import argparse
#import pickle
import os
import sys
import pathlib
from typing import Optional, Sequence, Callable
from dataclasses import dataclass, field

import ase
import numpy as np
import pandas as pd
#from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from md_interface_sim import folder_exist, build_pd
from md_interface_sim.plot_electrode_vs_wfunc import generalised_hydrogen_electrode, get_H_count, get_ion_count
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


def amanda_test() -> bool:
    if 'from_amanda' not in globals().keys(): return False
    return globals().get('from_amanda')



def main(dbs_dirs: Sequence[str], save_name, sim_names: Optional[Sequence[str]]=None, ph: float = 6, png: bool = False):

    dbs_dirs, dbs_selection = list(zip(*(db_dir.split('@') if '@' in db_dir else [db_dir, None] for db_dir in dbs_dirs)))
    for db_dir in dbs_dirs:
        if not os.path.basename(db_dir) in os.listdir(db_path if len(db_path := os.path.dirname(db_dir)) > 0 else '.'): raise FileNotFoundError("Can't find database")

    dat_pd = {name: build_pd(db_dir, select_key=sel_key) for name, db_dir, sel_key in zip(sim_names if sim_names is not None else dbs_dirs, dbs_dirs, dbs_selection)}

    ghe = lambda pd_series: generalised_hydrogen_electrode(
        E=pd_series['energy'],
        E_ref=dat_pd[sim_names[0] if sim_names is not None else dbs_dirs[0]]['energy'].mean(),
        n_proton=get_H_count(pd_series.get('atoms')),
        proton_pot=-6.616893 - (-0.49),  # ss
        cat_list=get_ion_count(pd_series.get('atoms')),
        work_func=pd_series['work_top' if not amanda_test() else 'wftop'],
        pH=ph,
        T=pd_series['temperature' if not amanda_test() else 'Temperature']
    )

    for db_name in dat_pd.keys():
        dat_pd[db_name].insert( dat_pd[db_name].shape[1], 'GHE', dat_pd[db_name].apply(ghe, axis=1))


    cov_bins_size = 0.02
    cov_bins_linSp = np.linspace(start=(bins_start := 2.5), stop=(bins_end := 6.5), num=int((bins_end - bins_start)/cov_bins_size))
    theta = []
    partition = []

    for pot in cov_bins_linSp:
        thata_temp = 0
        partition_temp = 0
        for db in dat_pd.values():
            if (db_pot := db.query('@pot < GHE > @pot + @cov_bins_size')).shape[0] > 0:
                name_later = np.exp(-db_pot['GHE'] / (0.000086173303*db_pot['temperature' if not amanda_test() else 'Temperature']))
                thata_temp += get_H_count(db.iloc[0].get('atoms')) * name_later
                partition_temp += name_later
        theta += [thata_temp]
        partition += [partition_temp]

    proton_coverage = [(the/par)/12 if par > 0 else 0 for the, par in zip(theta, partition)] # 12 is the number of surface atoms

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        mode='lines',
        y=[pro_cov for n in range(2) for pro_cov in proton_coverage],
        x=[cov_bin + n * cov_bins_size for n in range(2) for cov_bin in cov_bins_linSp],
        name='$H^*$',
        fill='tozeroy',
        line=dict(color='firebrick',),
        fillcolor='firebrick',
    ))

    fig.add_trace(go.Scatter(
        mode='lines',
        y=[-1*pro_cov for n in range(2) for pro_cov in proton_coverage],
        x=[cov_bin + n * cov_bins_size for n in range(2) for cov_bin in cov_bins_linSp],
        name='$OH^*$',
        fill='tozeroy',
        line=dict(color='blue', ),
        fillcolor='blue',
    ))

    fig.update_layout(
        #showlegend=False,
        xaxis_title=r'$\phi_{\rm e^-}$ [eV]',
        yaxis_title=r'$\Delta G_{\rm int}$ [eV]',
    )

    fig.update_xaxes(range=[bins_start, bins_end])
    fig.update_yaxes(range=[0, 0.55])

    folder_exist('plots')
    fig.write_html('plots/' + save_name + '.html', include_mathjax='cdn')
    if png: fig.write_image('plots/' + save_name + '.png')


if __name__ == '__main__':
    parser =argparse.ArgumentParser()
    parser.add_argument('db', type=str,  nargs='+', help='selection, can be added after @, note that the format have to follow the ase db select method. first database is reference!!!')
    parser.add_argument('save_name', type=str, help='name for the output file.')
    parser.add_argument('-names', '--sim_names', type=str, nargs='+', help='list of names to use for every database in the plot.')
    parser.add_argument('-ph', '--pH', type=float, default=6)
    parser.add_argument('--from_amanda', action='store_true')
    parser.add_argument('--png', action='store_true')
    args = parser.parse_args()

    global from_amanda
    from_amanda = args.from_amanda

    main(dbs_dirs=args.db,
         save_name=args.save_name,
         sim_names=args.sim_names,
         ph=args.pH,
         png=args.png)
