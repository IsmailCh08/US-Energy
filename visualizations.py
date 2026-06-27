"""
======================================================================
📊 COMPLETE VISUALIZATION BOILERPLATE (6 GRAPHS)
======================================================================
A reusable collection of 6 essential data visualization functions.
Just plug in your DataFrame and go!

HOW TO USE:
1. Copy this file into your project
2. Import: from visualizations import *
3. Call: plot_all_visualizations(df)  # to see everything!
4. Or call individual plots: plot_hourly_average(df)

CUSTOMIZE:
- Change 'PJME_MW' to your target column
- Change 'hour', 'day_of_week', etc. to your column names
- Adjust colors, sizes, and labels as needed

AUTHOR: Ismail Chaudry
DATE: 2026
======================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

# ============================================================
# STYLE SETTINGS
# ============================================================
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


# ============================================================
# PLOT 1: HOURLY AVERAGE (Bar Chart)
# ============================================================
def plot_hourly_average(df, value_col='PJME_MW', hour_col='hour'):
    """
    Plot 1: Hourly Average - Daily pattern.
    Best for: Seeing when demand peaks and drops during the day.
    """
    hourly_avg = df.groupby(hour_col)[value_col].mean()
    
    plt.figure(figsize=(14, 7))
    plt.bar(hourly_avg.index, hourly_avg.values, color=COLORS[0], 
            edgecolor='navy', alpha=0.8)
    
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel(f'Average {value_col}', fontsize=12)
    plt.title('Average by Hour', fontsize=14)
    plt.xticks(range(0, 24))
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return hourly_avg


# ============================================================
# PLOT 2: WEEKLY COMPARISON (Early vs Late)
# ============================================================
def plot_weekly_comparison(df, value_col='PJME_MW', week_col='week'):
    """
    Plot 2: Weekly Comparison - Early period vs Late period.
    Best for: Seeing if seasonal patterns changed over time.
    """
    df_early = df['2002':'2010']
    df_late = df['2010':'2018']
    
    weekly_avg_early = df_early.groupby(week_col)[value_col].mean()
    weekly_avg_late = df_late.groupby(week_col)[value_col].mean()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1.bar(weekly_avg_early.index, weekly_avg_early.values, color=COLORS[0])
    ax1.set_xlabel('Week of Year')
    ax1.set_ylabel(f'Average {value_col}')
    ax1.set_title('2002-2010')
    ax1.set_xticks(range(0, 53, 5))
    ax1.grid(True, axis='y', alpha=0.3)
    
    ax2.bar(weekly_avg_late.index, weekly_avg_late.values, color=COLORS[1])
    ax2.set_xlabel('Week of Year')
    ax2.set_ylabel(f'Average {value_col}')
    ax2.set_title('2010-2018')
    ax2.set_xticks(range(0, 53, 5))
    ax2.grid(True, axis='y', alpha=0.3)
    
    plt.suptitle('Weekly Comparison: Early vs Late Period', fontsize=14)
    plt.tight_layout()
    plt.show()


# ============================================================
# PLOT 3: WEEKDAY vs WEEKEND (Side by Side)
# ============================================================
def plot_weekday_vs_weekend(df, value_col='PJME_MW', hour_col='hour'):
    """
    Plot 3: Weekday vs Weekend Comparison.
    Best for: Seeing the difference in daily patterns between weekdays and weekends.
    """
    hourly_weekday = df.groupby(['weekend', hour_col])[value_col].mean().unstack()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Weekdays (weekend = False)
    weekday_data = hourly_weekday.loc[False]
    ax1.bar(weekday_data.index, weekday_data.values, color=COLORS[0])
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel(f'Average {value_col}')
    ax1.set_title('Weekdays (Mon-Fri)')
    ax1.set_xticks(range(0, 24))
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Weekends (weekend = True)
    weekend_data = hourly_weekday.loc[True]
    ax2.bar(weekend_data.index, weekend_data.values, color=COLORS[1])
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel(f'Average {value_col}')
    ax2.set_title('Weekends (Sat-Sun)')
    ax2.set_xticks(range(0, 24))
    ax2.grid(True, axis='y', alpha=0.3)
    
    plt.suptitle('Weekday vs Weekend by Hour', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    return hourly_weekday


# ============================================================
# PLOT 4: YEARLY AVERAGE (Bar Chart)
# ============================================================
def plot_yearly_average(df, value_col='PJME_MW', year_col='year'):
    """
    Plot 4: Yearly Average - Long-term trend.
    Best for: Seeing how demand has changed over the years.
    """
    yearly_avg = df.groupby(year_col)[value_col].mean()
    
    plt.figure(figsize=(14, 7))
    plt.bar(yearly_avg.index, yearly_avg.values, color=COLORS[2], 
            edgecolor='purple', alpha=0.8)
    
    plt.xlabel('Year', fontsize=12)
    plt.ylabel(f'Average {value_col}', fontsize=12)
    plt.title('Average by Year', fontsize=14)
    plt.xticks(yearly_avg.index, rotation=45)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return yearly_avg


# ============================================================
# PLOT 5: SEASONAL DECOMPOSITION
# ============================================================
def plot_seasonal_decomposition(df, value_col='PJME_MW', period=365):
    """
    Plot 5: Seasonal Decomposition - Trend, Seasonal, Residual.
    Best for: Understanding what's driving your data.
    
    Parameters:
    -----------
    df : DataFrame with datetime index
    value_col : str, column to decompose
    period : int, seasonal period (365 for daily, 7 for weekly, 12 for monthly)
    """
    # Resample to daily
    df_daily = df[value_col].resample('D').mean().dropna()
    
    # Decompose
    decomposition = seasonal_decompose(df_daily, model='additive', period=period)
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    
    # 1. Original
    axes[0].plot(decomposition.observed, color='black')
    axes[0].set_title('1. Original Data', fontsize=12)
    axes[0].set_ylabel(value_col)
    axes[0].grid(True, alpha=0.3)
    
    # 2. Trend
    axes[1].plot(decomposition.trend, color=COLORS[0])
    axes[1].set_title('2. Trend Component (Long-term Direction)', fontsize=12)
    axes[1].set_ylabel(value_col)
    axes[1].grid(True, alpha=0.3)
    
    # 3. Seasonal
    axes[2].plot(decomposition.seasonal, color=COLORS[1])
    axes[2].set_title('3. Seasonal Component (Repeating Patterns)', fontsize=12)
    axes[2].set_ylabel(value_col)
    axes[2].grid(True, alpha=0.3)
    
    # 4. Residual
    axes[3].plot(decomposition.resid, color=COLORS[3])
    axes[3].set_title('4. Residual Component (Random Noise)', fontsize=12)
    axes[3].set_xlabel('Date')
    axes[3].set_ylabel(value_col)
    axes[3].grid(True, alpha=0.3)
    
    plt.suptitle('Seasonal Decomposition', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    return decomposition


# ============================================================
# PLOT 6: CORRELATION HEATMAP
# ============================================================
def plot_correlation_heatmap(df, target_col='PJME_MW'):
    """
    Plot 6: Correlation Heatmap - Feature relationships.
    Best for: Seeing which features are correlated and which are redundant.
    """
    # Select numerical columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_df = df[numeric_cols]
    correlation_matrix = corr_df.corr()
    
    # Create heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Correlation Coefficient'},
        vmin=-1, vmax=1
    )
    plt.title('Feature Correlation Heatmap', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Print top correlations with target
    print("\n" + "=" * 50)
    print(f"🔍 Features most correlated with {target_col}:")
    print("=" * 50)
    target_corr = correlation_matrix[target_col].sort_values(ascending=False)
    for feature, corr in target_corr.items():
        if feature != target_col:
            print(f"  {feature:15} : {corr:+.3f}")
    print("=" * 50)
    
    return correlation_matrix


# ============================================================
# PLOT ALL: Master Function (6 Graphs)
# ============================================================
def plot_all_visualizations(df, value_col='PJME_MW'):
    """
    Run ALL 6 visualizations in sequence.
    Perfect for a complete data exploration report.
    """
    print("\n" + "=" * 60)
    print("📊 GENERATING ALL 6 VISUALIZATIONS")
    print("=" * 60)
    
    visualizations = [
        ("1. Hourly Average", plot_hourly_average),
        ("2. Weekly Comparison", plot_weekly_comparison),
        ("3. Weekday vs Weekend", plot_weekday_vs_weekend),
        ("4. Yearly Average", plot_yearly_average),
        ("5. Seasonal Decomposition", plot_seasonal_decomposition),
        ("6. Correlation Heatmap", plot_correlation_heatmap),
    ]
    
    for name, func in visualizations:
        print(f"\n▶ {name}...")
        try:
            func(df)
        except Exception as e:
            print(f"   ⚠️ Error: {e}")
    
    print("\n" + "=" * 60)
    print("✅ ALL 6 VISUALIZATIONS COMPLETE!")
    print("=" * 60)


# ============================================================
# QUICK REFERENCE CARD (Updated with 6 Graphs)
# ============================================================
"""
┌─────────────────────────────────────────────────────────────────┐
│                    QUICK REFERENCE CARD (6 GRAPHS)            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  plot_hourly_average(df)          # Daily pattern              │
│  plot_weekly_comparison(df)       # Early vs Late              │
│  plot_weekday_vs_weekend(df)      # Weekday vs Weekend         │
│  plot_yearly_average(df)          # Yearly trend               │
│  plot_seasonal_decomposition(df)  # Trend, Season, Noise      │
│  plot_correlation_heatmap(df)     # Feature relationships      │
│                                                                 │
│  plot_all_visualizations(df)      # Run ALL 6 at once!        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
"""


# ============================================================
# USAGE EXAMPLE
# ============================================================
if __name__ == "__main__":
    from src.data_loader import load_data, prepare_data
    
    # Load your data
    df = load_data("Energy_Data.csv")
    df = prepare_data(df)
    
    # Option 1: Run all 6 visualizations at once
    plot_all_visualizations(df)
    
    # Option 2: Run individual plots
    # plot_hourly_average(df)
    # plot_weekly_comparison(df)
    # plot_weekday_vs_weekend(df)
    # plot_yearly_average(df)
    # plot_seasonal_decomposition(df)
    # plot_correlation_heatmap(df)