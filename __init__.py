import os
import time
from typing import NoReturn, Sequence, Tuple, Optional
from dataclasses import dataclass, field
from contextlib import contextmanager
import traceback
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed
import ase.db as db
import pandas as pd
from sqlite3 import OperationalError


def build_pd(db_dir_list, select_key: Optional = None):
    if isinstance(db_dir_list, str): db_dir_list = [db_dir_list]
    for db_dir in db_dir_list:
        if not os.path.basename(db_dir) in os.listdir(db_path if len(db_path := os.path.dirname(db_dir)) > 0 else '.'):
            raise FileNotFoundError("Can't find database")
    db_list = [db.connect(work_db) for work_db in db_dir_list]
    pd_dat = pd.DataFrame([row.__dict__ for work_db in db_list for row in work_db.select(selection=select_key)])
    return pd_dat
