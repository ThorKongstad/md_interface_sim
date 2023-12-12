import argparse
from typing import Sequence, NoReturn, Tuple, Iterable, Optional
import os
import time
import sys
import pathlib
from subprocess import call

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from md_interface_sim import build_pd, folder_exist

import ase.db as db
import plotly.graph_objects as go


def plot_bins_work_func(panda_data, save_name: str):
    binsize = 0.02 # this is not the correct way to do stuff, but it is the fast way.
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=panda_data['work_top'],
        xbins=dict(size=binsize)
    ))

    fig.update_layout(
        showlegend=False,
        xaxis_title='\Phi',
        yaxis_title='Image count',
    )

    folder_exist('plots')
    fig.write_html('plots/' + save_name + '.html', include_mathjax='cdn')



def main(md_db: str):
    md_pd = build_pd(md_db)

    plot_name = os.path.basename(md_db).replace('.db', '') + '_bin_plot'

    plot_bins_work_func(md_pd, plot_name)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('db', type=str)
    args = parser.parse_args()
