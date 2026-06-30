from src.data_loader import load_data, prepare_data
from src.preprocessing import split_time_series
from src.model import (
    create_linear_regression_pipeline,
    train_model,
    evaluate_model,
    create_random_forest_pipeline,
    create_XGBoost_pipeline
)

def main():
    print("=" * 60)
    print("Energy Prediction Model")
    print("=" * 60)
    
    # 1. Load and prepare
    print("\n Loading data...")
    df = load_data("Energy_Data.csv")
    df = prepare_data(df)
    
    # 2. Split
    print("\n Splitting data...")
    X_train, y_train, X_test, y_test = split_time_series(df, target_col='PJME_MW', test_ratio=0.2)
    print(f"   Training: {X_train.shape[0]} rows")
    print(f"   Test: {X_test.shape[0]} rows")
    
    # 3. Create and train Linear Regression pipeline
    print("Training Linear Regression...")
    lr_pipeline = create_linear_regression_pipeline()
    lr_pipeline = train_model(lr_pipeline, X_train, y_train)
    lr_results = evaluate_model(lr_pipeline, X_test, y_test)
    print(f"   RMSE: {lr_results['rmse']:.2f}")
    print(f"   MAE:  {lr_results['mae']:.2f}")
    print(f"   R²:   {lr_results['r2']:.4f}")

    # Create and train Random Forest Pipeline
    print("Training Random Forest...")
    rf_pipeline = create_random_forest_pipeline()
    rf_pipeline = train_model(rf_pipeline, X_train, y_train)
    rf_results = evaluate_model(rf_pipeline, X_test, y_test)
    print(f"   RMSE: {rf_results['rmse']:.2f}")
    print(f"   MAE:  {rf_results['mae']:.2f}")
    print(f"   R²:   {rf_results['r2']:.4f}")

    # Create and train XGBoost pipeline
    print("Training XGBoost...")
    xg_pipeline = create_XGBoost_pipeline()
    xg_pipeline = train_model(xg_pipeline, X_train, y_train)
    xg_results = evaluate_model(xg_pipeline, X_test, y_test)
    print(f"   RMSE: {xg_results['rmse']:.2f}")
    print(f"   MAE:  {xg_results['mae']:.2f}")
    print(f"   R²:   {xg_results['r2']:.4f}")
    
    # 5. Print results
    print("\n" + "-" * 60)
    print("Model Comparison")
    print("-" * 60)
    print(f" Linear Regression:")
    print(f"   RMSE: {lr_results['rmse']:.2f}")
    print(f"   R²:   {lr_results['r2']:.4f}")
    print(f" Random Forest:")
    print(f"   RMSE: {rf_results['rmse']:.2f}")
    print(f"   R²:   {rf_results['r2']:.4f}")
    print(f" XGBoost: ")
    print(f"   RMSE: {xg_results['rmse']:.2f}")
    print(f"   R²:   {xg_results['r2']:.4f}")

    improvement = ((rf_results['rmse']- xg_results['rmse'])/ xg_results['rmse']) * 100
    print(f"XGBoost improves RMSE by {improvement:.1f}%")
    print("=" * 60)

if __name__ == "__main__":
    main()

    
