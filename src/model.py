from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from .data_loader import load_data, prepare_data
from .preprocessing import split_time_series
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV

# process and standarize data
def create_preprocessor():
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), ['hour', 'day_of_week', 'month', 'quarter', 'year'])
    ])
    return preprocessor

# Create linear regression pipeline
def create_linear_regression_pipeline():

    preprocessor = create_preprocessor()
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', LinearRegression())
    ])
    return pipeline

def create_random_forest_pipeline(n_estimators=200, max_depth=10, random_state=42):
    preprocessor = create_preprocessor()
    pipepline = Pipeline([
        ('preprocessor',preprocessor),
        ('model', RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, max_depth=max_depth))
    ])
    return pipepline

def create_XGBoost_pipeline(n_estimators=150, learning_rate=0.05, random_state=42, max_depth=4):
    preprocessor = create_preprocessor()
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', XGBRegressor(n_estimators=n_estimators,learning_rate=learning_rate,random_state=random_state, max_depth=max_depth))
    ])
    return pipeline

def tune_XGBoost(X_train,y_train):

    preprocessor = create_preprocessor()

    param_grid = {
        'model__n_estimators': [100, 200, 300],
        'model__learning_rate': [0.01, 0.05, 0.1],
        'model__max_depth': [3, 5, 7],
        'model__reg_alpha': [0, 0.1, 1],
        'model__reg_lambda': [1, 5, 10]
    }

    pipeline = Pipeline([
        ('preprocessor', create_preprocessor()),
        ('model', XGBRegressor(random_state=42, verbosity=0))
    ])

    random_search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_grid,
    n_iter=20,  # Try 20 random combos instead of all
    cv=5,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    random_state=42
)

    random_search.fit(X_train, y_train)
    
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best RMSE (cross-validated): {-random_search.best_score_:.2f}")
    
    return random_search.best_estimator_


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
