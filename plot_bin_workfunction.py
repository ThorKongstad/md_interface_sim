import argparse
from typing import Sequence, NoReturn, Tuple, Iterable, Optional
import os
import time
import sys
import pathlib
from subprocess import call
#from operator import itemgetter

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from md_interface_sim import build_pd, folder_exist

import numpy as np
from scipy.stats import norm
from pandas import DataFrame
import ase.db as db
import plotly.graph_objects as go
from numpy import histogram_bin_edges
from scipy.stats import norm, goodness_of_fit


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
        y=(panda_data['kinitic_E' if not amanda_test() else 'Ekin'] * (2/3)) / (0.000086173303*free_atoms) * len(atoms)/free_atoms, #(at.get_temperature() for at in panda_data['atoms'].iloc)
    ))

    fig.update_layout(
        yaxis_title=r'Temperature (K)',
        xaxis_title='Id nr',
    )

    return fig


def plot_residual_energy(panda_data: DataFrame,) -> go.Figure:
    fig = go.Figure()

    residual = lambda index: norm.fit(panda_data['energy'].iloc[index:] + panda_data['kinitic_E' if not amanda_test() else 'Ekin'].iloc[index:])[1]
    residual_gen = map(residual, range(panda_data.shape[0]))

    fig.add_trace(go.Scatter(
        mode='markers',
        name='energy residual',
        x=panda_data['id'],
        y=tuple(residual_gen),
    ))

    fig.update_layout(
        yaxis_title=r'e_i - <E[i:]>',
        xaxis_title='Id nr',
    )

    return fig


def plot_3d_hist(panda_data: DataFrame,) -> go.Figure:
    bin_edges_x = histogram_bin_edges(panda_data['work_top' if not amanda_test() else 'wftop'].dropna(), bins='fd')
    binsize_x = bin_edges_x[1] - bin_edges_x[0]

    bin_edges_y = histogram_bin_edges(panda_data['time' if not amanda_test() else 'id'].dropna(), bins='fd')
    binsize_y = bin_edges_y[1] - bin_edges_y[0]

    fig = go.Figure()
    fig.add_trace(go.Histogram2d(
        x=panda_data['work_top' if not amanda_test() else 'wftop'],
        y=panda_data['time' if not amanda_test() else 'id'],
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


def plot_Wfunc_deviation(panda_data: DataFrame,) -> go.Figure:
    fig = go.Figure()

    residual_back = lambda index: norm.fit(panda_data['work_top' if not amanda_test() else 'wftop'].iloc[:index])
    residual_forward = lambda index: norm.fit(panda_data['work_top' if not amanda_test() else 'wftop'].iloc[index:])
    residual_back_gen = list(map(residual_back, range(panda_data.shape[0])))
    residual_forward_gen = list(map(residual_forward, range(panda_data.shape[0])))

    upper_bound_back = [res_norm[0] + res_norm[1] for res_norm in residual_back_gen]
    lower_bound_back = [res_norm[0] - res_norm[1] for res_norm in residual_back_gen]

    upper_bound_forward = [res_norm[0] + res_norm[1] for res_norm in residual_forward_gen]
    lower_bound_forward = [res_norm[0] - res_norm[1] for res_norm in residual_forward_gen]

    fig.add_trace(go.Scatter(
        mode='lines',
        y=[res_norm[0] for res_norm in residual_back_gen],
        x=panda_data['id'],
    ))

    fig.add_trace(go.Scatter(
        #mode='lines',
        x=panda_data['id'].to_list() + panda_data['id'].iloc[::-1].to_list(),
        y=upper_bound_back + list(reversed(lower_bound_back)),
        hoverinfo="skip",
        showlegend=False,
        fill='toself',
        fillcolor='rgba(101,118,164,0.2)'#'rgba(0,100,80,0.2)',
    ))

    fig.add_trace(go.Scatter(
        #mode='lines',
        x=panda_data['id'].to_list() + panda_data['id'].iloc[::-1].to_list(),
        y=upper_bound_forward + list(reversed(lower_bound_forward)),
        hoverinfo="skip",
        showlegend=False,
        fill='toself',
        fillcolor='rgba(214,83,31,0.2)',
    ))

    fig.update_layout(
        yaxis_title=r'<$\Phi$[:i]>',
        xaxis_title='id no.',
    )

    return fig


def plot_fit_goodness(panda_data: DataFrame,) -> go.Figure:
    fig = go.Figure()

    fit_test_forward = lambda index: goodness_of_fit(
        dist=norm,
        data=panda_data['work_top' if not amanda_test() else 'wftop'].iloc[index:],
        statistic='ks'
    )
    fit_test_forward_gen = list(map(fit_test_forward, range(panda_data.shape[0]-1)))

    fig.add_trace(go.Scatter(
        mode='lines',
        y=[fit_test.statistic for fit_test in fit_test_forward_gen],
        x=panda_data['id'],
    ))

    fig.update_layout(
        yaxis_title=r'ks stats norm(<$\Phi$[:i]>)',
        xaxis_title='id no.',
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
        histnorm= "probability density",# "" | "percent" | "probability" | "density" | "probability density"
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
        yaxis_title='Image count density',
        title=f'Count: {len(panda_data.index)}'
    )

    fig.set_subplots(rows=3, cols=2,
                     specs=[[{}, {}],
                            [{"colspan": 2}, None],
                            [{"colspan": 2}, None]],
                     row_heights=[0.5, 0.25, 0.25])

    #fig.add_traces(data=(Wfunc_plot := plot_Wfunc_deviation(panda_data)).data, rows=1, cols=2)
    # fig.update_layout(hist3d_plot.to_dict()['layout'], row=1, col=2)
    #fig.update_xaxes(title_text=Wfunc_plot.layout.xaxis.title.text, row=1, col=2)
    #fig.update_yaxes(title_text=Wfunc_plot.layout.yaxis.title.text, row=1, col=2)

    fig.add_traces(data=(fit_plot := plot_fit_goodness(panda_data)).data, rows=1, cols=2)
    # fig.update_layout(hist3d_plot.to_dict()['layout'], row=1, col=2)
    fig.update_xaxes(title_text=fit_plot.layout.xaxis.title.text, row=1, col=2)
    fig.update_yaxes(title_text=fit_plot.layout.yaxis.title.text, row=1, col=2)

    fig.add_traces(data=(T_plot := plot_temperature(panda_data)).data, rows=2, cols=1)
    #fig.update_layout(T_plot.to_dict()['layout'], row=2, col=1)
    fig.update_xaxes(title_text=T_plot.layout.xaxis.title.text, row=2, col=1)
    fig.update_yaxes(title_text=T_plot.layout.yaxis.title.text, row=2, col=1)

    #fig.add_trace(*(hist3d_plot := plot_3d_hist(panda_data)).data, row=1, col=2)
    #fig.update_layout(hist3d_plot.to_dict()['layout'], row=1, col=2)
    #fig.update_xaxes(title_text=hist3d_plot.layout.xaxis.title.text, row=1, col=2)
    #fig.update_yaxes(title_text=hist3d_plot.layout.yaxis.title.text, row=1, col=2)

    fig.add_traces(data=(residual_plot := plot_residual_energy(panda_data)).data, rows=3, cols=1)
    fig.update_xaxes(title_text=residual_plot.layout.xaxis.title.text, row=3, col=1)
    fig.update_yaxes(title_text=residual_plot.layout.yaxis.title.text, row=3, col=1)

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
