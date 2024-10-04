import argparse
import pickle
import os
import sys
import pathlib
from typing import Optional
from dataclasses import dataclass, field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from md_interface_sim import folder_exist, build_pd
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


def main(db_dir: str):
    if not os.path.basename(db_dir) in os.listdir(db_path if len(db_path := os.path.dirname(db_dir)) > 0 else '.'): raise FileNotFoundError("Can't find database")

    md_panda = build_pd(db_dir)

    fig = px.line(
        data_frame=md_panda,
        x='work_top',
        y='time',
        markers=True
    )

    fig.update_layout(xaxis_title='Work function', yaxis_title='md time')

    folder_exist(f'misc_plots')
    save_name = f'misc_plots/' + os.path.basename(db_dir).split('.')[0] + '_path' #f'workfunc_plots_{calculation_name}/time_{time_step}'
    fig.write_html(save_name + '.html', include_mathjax='cdn')


if __name__ == '__main__':
    parser =argparse.ArgumentParser()
    parser.add_argument('db', type=str)
#    parser.add_argument('row_index', type=int)
    args = parser.parse_args()

    main(db_dir=args.db)
