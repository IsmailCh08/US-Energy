import pandas as pd
import numpy as np
from src.data_loader import load_data, prepare_data
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose


df = load_data("Energy_Data.csv")
df = df.sort_index()
df = prepare_data(df)




# Load and prepare data
df = load_data("Energy_Data.csv")
df = prepare_data(df)

# For decomposition, you need to resample to daily or monthly
# (Decomposition works better with regular frequency)

# OPTION 1: Daily resampling (recommended for this dataset)
df_daily = df['PJME_MW'].resample('D').mean().dropna()

# OPTION 2: Weekly resampling (smoother)
# df_weekly = df['PJME_MW'].resample('W').mean().dropna()

# Perform decomposition (choose period = 365 for daily data, 52 for weekly)
decomposition = seasonal_decompose(df_daily, model='additive', period=365)

# Create the plot
fig, axes = plt.subplots(4, 1, figsize=(14, 12))

# 1. Original data
axes[0].plot(decomposition.observed, color='black')
axes[0].set_title('Original Data')
axes[0].set_ylabel('Demand (MW)')
axes[0].grid(True, alpha=0.3)

# 2. Trend component
axes[1].plot(decomposition.trend, color='blue')
axes[1].set_title('Trend Component (Long-term Direction)')
axes[1].set_ylabel('Demand (MW)')
axes[1].grid(True, alpha=0.3)

# 3. Seasonal component
axes[2].plot(decomposition.seasonal, color='green')
axes[2].set_title('Seasonal Component (Repeating Patterns)')
axes[2].set_ylabel('Demand (MW)')
axes[2].grid(True, alpha=0.3)

# 4. Residual component (noise)
axes[3].plot(decomposition.resid, color='red')
axes[3].set_title('Residual Component (Random Noise)')
axes[3].set_xlabel('Date')
axes[3].set_ylabel('Demand (MW)')
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()