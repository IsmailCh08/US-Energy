import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error

df = pd.read_csv("Energy_Data.csv")

print(df.head())