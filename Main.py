import pandas as pd
import numpy as np
from src.data_loader import load_data, prepare_data


df = load_data("Energy_Data.csv")
df = df.sort_index()
df = prepare_data(df)
