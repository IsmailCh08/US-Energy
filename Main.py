import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from data_loader import load_data, prepare_data


df = load_data("Energy_Data.csv")
df = df.sort_index()
df = prepare_data(df)

    
