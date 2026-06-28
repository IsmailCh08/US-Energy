from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from data_loader import load_data, prepare_data
from preprocessing import split_time_series

df = load_data("Energy_Data.csv")
df = prepare_data(df)
X_train, y_train, X_test, y_test = split_time_series(df,target_col='PJME_MW', test_ratio=0.2)

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), ['hour', 'day_of_week', 'month','quarter','year']),
    ('cat', OneHotEncoder(), ['is_weekend'])
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model',LinearRegression())
])


pipeline.fit(X_train,y_train)
predictions = pipeline.pred(X_test)

rmse = root_mean_squared_error(y_test, predictions)
print(f"RMSE: {rmse:.2f}")