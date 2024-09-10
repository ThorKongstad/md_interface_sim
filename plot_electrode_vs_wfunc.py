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


@dataclass
class Ion:
    n: int
    name: str
    E: float = None
    colour: str = None

    def __post_init__(self):
        if self.E is None:
            E_H2 = -6.635# -6.616893  # RPBE lcao calculation with Cu ghost
            E_H2O = -13.528367  # RPBE lcao calculation with Cu ghost
            G_H2 = E_H2-(-0.49)
            match self.name:
                case 'Na':
                    E_Na2O_s = -36.713522 / 4  # eight Na atoms in unit cell
                    G_Na_s = 0.7764758890
                    self.E = 0.5*(E_Na2O_s - E_H2O + E_H2 + G_H2) + G_Na_s
                case 'K':
                    E_K2O2_s = -41.970807 / 4  # eight K atoms in unit cell
                    G_K_s = 0.239313298
                    self.E = 0.5*(E_K2O2_s - 2*E_H2O + 2*(E_H2 + G_H2)) + G_K_s
                #case 'Li':
                    #E_Li2O_s = -47.736909 / 4  # eight Li atoms in unit cell
                    #G_Li_s = 1.875098276
                    #self.E =

        if self.colour is None:
            match self.name:
                case 'Na':
                    self.colour = 'rgb(171, 92, 243)' #jmol
                case 'K':
                    self.colour = 'rgb(143, 64, 212)'


def generalised_hydrogen_electrode(E: float, E_ref: float, n_proton: float, proton_pot: float, cat_list: Sequence[Ion], work_func: float, pH: float, T: float): return E - E_ref - sum(ion.n * ion.E for ion in cat_list) - 0.5*n_proton*proton_pot - n_proton*(4.4 - work_func - 2.303 * (8.617*10**-5)*T*pH)


def mean(values: Sequence[float]) -> float: return sum(values) / len(values)


def make_trace(name, db: pd.DataFrame, ghe_lambda: Callable[[pd.Series], float], Hcolor_fraction: Optional[float] = None):
    ion_list = get_ion_count(db.iloc[0].get('atoms'))

    Hcoler_args = dict(
            #colorscale='RdBu',
            color=px.colors.sample_colorscale(colorscale='RdBu', samplepoints=Hcolor_fraction)[0],
    ) if Hcolor_fraction is not None else dict()

    return go.Scatter(
        name=name,
        x=(x_val:= db['work_top' if not amanda_test() else 'wftop']),
        y=db.apply(ghe_lambda, axis=1),
        meta=dict(xmean=mean(x_val), colour_scale='RdBu'),
        mode='markers',
        hovertemplate='mean: %{meta.xmean:.2f}',
        marker=dict(
#            opacity=0.7,
            line=dict(color=('DarkSlateGrey' if len(ion_list) == 0 else ion_list[0].colour), width=1),
            **Hcoler_args
        ),
    )


def get_H_count(atoms: ase.Atoms,) -> int:
    return atoms.get_chemical_formula(mode='all').count('H') - 64


def get_ion_count(atoms: ase.Atoms):
    return tuple(Ion(n=atoms.get_chemical_formula(mode='all').count(ion), name=ion) for ion in ('K', 'Na') if ion in atoms.get_chemical_formula(mode='all'))


def amanda_test() -> bool:
    if 'from_amanda' not in globals().keys(): return False
    return globals().get('from_amanda')


def sp(x):
    print(x, type(x))
    return x


def main(dbs_dirs: Sequence[str], save_name, sim_names: Optional[Sequence[str]] = None, ph: float = 6, png: bool = False, Hcolor_bool: bool = False):
    dbs_dirs, dbs_selection = list(zip(*(db_dir.split('@') if '@' in db_dir else [db_dir, None] for db_dir in dbs_dirs)))
    for db_dir in dbs_dirs:
        if not os.path.basename(db_dir) in os.listdir(db_path if len(db_path := os.path.dirname(db_dir)) > 0 else '.'): raise FileNotFoundError("Can't find database")

    dat_pd = {name: build_pd(db_dir, select_key=sel_key) for name, db_dir, sel_key in zip(sim_names if sim_names is not None else dbs_dirs, dbs_dirs, dbs_selection)}

    ghe = lambda pd_series: generalised_hydrogen_electrode(
        E=pd_series['energy'],
        E_ref=dat_pd[sim_names[0] if sim_names is not None else dbs_dirs[0]]['energy'].mean(),
        n_proton=get_H_count(pd_series.get('atoms')),
        proton_pot=-6.635-(-0.49),#-6.616893-(-0.49), #ss
        cat_list=get_ion_count(pd_series.get('atoms')),
        work_func=pd_series['work_top' if not amanda_test() else 'wftop'],
        pH=ph,
        T=pd_series['temperature' if not amanda_test() else 'Temperature']
    )

    if Hcolor_bool:
        proton_counts = [get_H_count(db.iloc[0].get('atoms')) for db in dat_pd.values()]
        max_pro_devi = max(proton_counts, key=abs)

    fig = go.Figure()
    for key, val in dat_pd.items():
        fig.add_trace(make_trace(
            name=key,
            db=val,
            ghe_lambda=ghe,
            Hcolor_fraction=((0.5 + (get_H_count(val.iloc[0].get('atoms'))/max_pro_devi) * 0.5) if Hcolor_bool else None),
        ))

    fig.update_layout(
#        showlegend=False,
        xaxis_title=r'$\Phi$',
        yaxis_title=r'$E_{GCHE}$',
#        height=1392,
#        width=697.92,
    )

    for trace in fig.data:
        fig.add_vline(
            x=trace.meta.get('xmean'),
            line_dash='dash',
            line_color=trace.marker.color,
            line_width=3,
            label=dict(
                text=trace.name,
                textposition="start",
                textangle=90,
                padding=30,
                font=dict(size=15)
            )
        )

    #fig.update_xaxes(range=[-4, +4])
    fig.update_yaxes(range=[-20, +20])

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
    parser.add_argument('--Hcolour', action='store_true', help='bool for colouring after no. of H')
    args = parser.parse_args()

    global from_amanda
    from_amanda = args.from_amanda

    main(dbs_dirs=args.db,
         save_name=args.save_name,
         sim_names=args.sim_names,
         ph=args.pH,
         png=args.png,
         Hcolor_bool=args.Hcolour)
