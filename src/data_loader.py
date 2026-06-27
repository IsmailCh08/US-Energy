import pandas as pd
from pathlib import Path

def load_data(file_path):
    
    # heck if file exists
    if not Path(file_path).exists():
        print(f" File not found: {file_path}")
        return None 
    
    try:
        # Load the CSV
        df = pd.read_csv(file_path)
        print(f"Raw data loaded: {len(df)} rows")
        
        # Clean column names
        df.columns = df.columns.str.strip()
        print(f"Columns cleaned: {df.columns.tolist()}")
        
        # Parse dates
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        print(f"Dates parsed")
        
        # Set as index
        df = df.set_index('Datetime')
        print(f"Index set")
        
        # RETURN the DataFrame
        return df  
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def prepare_data(df):

    df['hour'] = df.index.hour
    df['is_morning']= (df['hour'] >= 6) & (df['hour'] <= 11)
    df['is_afternoon'] = (df['hour'] >= 12) & (df['hour'] <= 17)
    df['is_evening'] = (df['hour'] >= 18) & (df['hour'] < 22)
    df['is_night'] = (df['hour'] >= 22) | (df['hour'] < 6)


    df['day_of_week'] = df.index.dayofweek
    df['weekend'] = df['day_of_week'] >= 5
    df['weekday'] = df['day_of_week'] < 5

    df['month'] = df.index.month
    df['year'] =  df.index.year
    df['quarter'] = df.index.quarter
    df['day_of_year'] = df.index.dayofyear
    df['week'] = df.index.isocalendar().week

    return df
    


