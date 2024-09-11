import argparse
from typing import Sequence, NoReturn, Tuple, Iterable, Optional
import os
import time
import sys
import pathlib
from subprocess import call

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from md_interface_sim import build_pd, folder_exist

import numpy as np
from scipy.stats import norm
from pandas import DataFrame
import ase.db as db
import plotly.graph_objects as go
from numpy import histogram_bin_edges
from scipy.stats import norm


def amanda_test() -> bool:
    if 'from_amanda' not in globals().keys(): return False
    return globals().get('from_amanda')


def plot_temperature(panda_data: DataFrame,) -> go.Figure:
    fig = go.Figure()

    atoms = panda_data['atoms'].iloc[0]
    free_atoms = len(atoms) - sum([len(con.index) for con in atoms.constraints])

    fig.add_trace(go.Scatter(
        mode='markers',
        name='Temperature' if not amanda_test() else 'Temperature',
        x=panda_data['id'],
        y=(panda_data['kinitic_E' if not amanda_test() else 'Ekin'] * (2/3)) / (0.000086173303*free_atoms) * len(atoms)/free_atoms,
    ))

    fig.update_layout(
        xaxis_title=r'Temperature (K)',
        yaxis_title='Id nr',
    )

    return fig

def plot_residual_energy(panda_data: DataFrame,) -> go.Figure:
    fig = go.Figure()

    residual = lambda index: norm.fit(panda_data['energy'].iloc[index:] + panda_data['kinitic_E' if not amanda_test() else 'Ekin'].iloc[index:])[1]
    residual_gen = map(residual, range(panda_data.shape[0]))

    fig.add_trace(go.Scatter(
        mode='markers',
        name='Temperature' if not amanda_test() else 'Temperature',
        x=panda_data['id'],
        y=tuple(residual_gen),
    ))

    fig.update_layout(
        xaxis_title=r'e_i - <E[i:]>',
        yaxis_title='Id nr',
    )

    return fig


def plot_3d_hist(panda_data: DataFrame,) -> go.Figure:
    bin_edges_x = histogram_bin_edges(panda_data['work_top' if not amanda_test() else 'wftop'].dropna(), bins='fd')
    binsize_x = bin_edges_x[1] - bin_edges_x[0]

    bin_edges_y = histogram_bin_edges(panda_data['time'].dropna(), bins='fd')
    binsize_y = bin_edges_y[1] - bin_edges_y[0]

    fig= go.Figure()
    fig.add_trace(go.Histogram2d(
        x=panda_data['work_top' if not amanda_test() else 'wftop'],
        y=panda_data['time'],
        histnorm="probability density",
        xbins=dict(
            start=bin_edges_x[0],
            size=binsize_x,
            end=bin_edges_x[-1]
        ),
        ybins=dict(
            start=bin_edges_y[0],
            size=binsize_y,
            end=bin_edges_y[-1]
        ),
    ))

    fig.update_layout(
        xaxis_title=r'$\Phi$',
        yaxis_title='time',
    )

    return fig


def plot_bins_work_func(panda_data: DataFrame, save_name: str, png: bool):
    bin_edges = histogram_bin_edges(panda_data['work_top' if not amanda_test() else 'wftop'].dropna(), bins='fd')
    binsize = bin_edges[1] - bin_edges[0]
    #binsize = 0.02 # this is not the correct way to do stuff, but it is the fast way.

    mu_fit, sd_fit = norm.fit(panda_data['work_top' if not amanda_test() else 'wftop'].dropna())

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=panda_data['work_top' if not amanda_test() else 'wftop'],
        xbins=dict(
            start=bin_edges[0],
            size=binsize,
            end=bin_edges[-1]
        ),
        histnorm= "probability density", # "" | "percent" | "probability" | "density" | "probability density"
    ))

    line = np.linspace(panda_data['work_top' if not amanda_test() else 'wftop'].min(), panda_data['work_top' if not amanda_test() else 'wftop'].max(), 1000)

    fig.add_trace(go.Scatter(
        x=line,
        y=tuple(map(lambda x: 1/(sd_fit*np.sqrt(2*np.pi))*np.exp(-0.5*((x-mu_fit)/sd_fit)**2), line)),
        name=f'norm fit',
        hovertemplate=f'norm: mu={mu_fit:.2f}, sd={sd_fit:.2f}'
    ))

    fig.update_layout(
        showlegend=False,
        xaxis_title=r'$\Phi$',
        yaxis_title='Image count',
        title=f'Count: {len(panda_data.index)}'
    )

    fig.set_subplots(rows=3, cols=2,
                     specs=[[{}, {}],
                            [{"colspan": 2}, None],
                            [{"colspan": 2}, None]],
                     row_heights=[0.5, 0.25, 0.25])

    fig.add_trace(*(T_plot := plot_temperature(panda_data)).data, row=2, col=1)
    #fig.update_layout(T_plot.to_dict()['layout'], row=2, col=1)
    fig.update_xaxes(title_text=T_plot.layout.__dict__.get('xaxis_title'), row=2, col=1)
    fig.update_yaxes(title_text=T_plot.layout.__dict__.get('yaxis_title'), row=2, col=1)

    fig.add_trace(*(hist3d_plot := plot_3d_hist(panda_data)).data, row=1, col=2)
    #fig.update_layout(hist3d_plot.to_dict()['layout'], row=1, col=2)
    fig.update_xaxes(title_text=hist3d_plot.layout.__dict__.get('xaxis_title'), row=1, col=2)
    fig.update_yaxes(title_text=hist3d_plot.layout.__dict__.get('yaxis_title'), row=1, col=2)

    fig.add_trace(*(residual_plot := plot_residual_energy(panda_data)).data, row=3, col=1)
    fig.update_xaxes(title_text=residual_plot.layout.__dict__.get('xaxis_title'), row=3, col=1)
    fig.update_yaxes(title_text=residual_plot.layout.__dict__.get('yaxis_title'), row=3, col=1)

    folder_exist('plots')
    fig.write_html('plots/' + save_name + '.html', include_mathjax='cdn')
    if png: fig.write_image('plots/' + save_name + '.png')


def main(md_db: str, png: bool):
    db_dir, db_selection = md_db.split('@') if '@' in md_db else [md_db, None]
    md_pd = build_pd(db_dir, select_key='time>0' if db_selection is None else db_selection)

    plot_name = os.path.basename(db_dir).replace('.db', '') + '_bin_plot'

    plot_bins_work_func(md_pd, plot_name, png)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('db', type=str,  help='selection, can be added after @, note that the format have to follow the ase db select method.')
    parser.add_argument('--png', action='store_true')
    parser.add_argument('--from_amanda', action='store_true')
    args = parser.parse_args()

    global from_amanda
    from_amanda = args.from_amanda

    main(args.db, args.png)
