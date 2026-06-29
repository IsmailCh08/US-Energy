from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from .data_loader import load_data, prepare_data
from .preprocessing import split_time_series
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score


# load and prepare data
df = load_data("Energy_Data.csv")
df = prepare_data(df)
X_train, y_train, X_test, y_test = split_time_series(df,target_col='PJME_MW', test_ratio=0.2)


def create_preprocessor():
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), ['hour', 'day_of_week', 'month', 'quarter', 'year'])
    ])
    return preprocessor

def create_linear_regression_pipeline():

    preprocessor = create_preprocessor()
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', LinearRegression())
    ])
    return pipeline

def create_random_forest_pipeline(n_estimators=100):
    preprocessor = create_preprocessor()
    pipepline = Pipeline([
        ('preprocessor',preprocessor),
        ('model', RandomForestRegressor(n_estimators=n_estimators, random_state=42))
    ])
    return pipepline


def train_model(pipeline, X_train, y_train):

    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(pipeline, X_test, y_test):

    predictions = pipeline.predict(X_test)
    
    rmse = root_mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': predictions
    }
