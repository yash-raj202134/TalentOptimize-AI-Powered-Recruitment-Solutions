# src/utils/data_loader.py

import pandas as pd # type: ignore
from src.constants.paths import DATA_PATH

def load_data():
    data = pd.read_csv(DATA_PATH)
    return data
