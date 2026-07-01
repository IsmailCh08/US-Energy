from src.data_loader import load_data, prepare_data
import pandas as pd
import matplotlib.pyplot as plt
from src.preprocessing import split_time_series
import joblib
import torch
from src.model import (
    create_linear_regression_pipeline,
    train_model,
    evaluate_model,
    create_random_forest_pipeline,
    create_XGBoost_pipeline,
    tune_XGBoost
)
from src.deeplearning import LTSMModel, prepare_lstm_data, train_lstm, evaluate_lstm

def main():
    print("=" * 60)
    print("Energy Prediction Model")
    print("=" * 60)
    
    # Load and prepare
    print("\n Loading data...")
    df = load_data("Energy_Data.csv")
    df = prepare_data(df)
    
    # Split
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

    # Create and Tune XGBoost pipeline
    print("Tuning XGBoost with Grid Search + Cross-Validation...")
    best_xgb_pipeline = tune_XGBoost(X_train, y_train)

    xg_results = evaluate_model(best_xgb_pipeline, X_test, y_test)
    print(f"Tuned XGBoost Results:")
    print(f"   RMSE: {xg_results['rmse']:.2f}")
    print(f"   MAE:  {xg_results['mae']:.2f}")
    print(f"   R²:   {xg_results['r2']:.4f}")


    # LSTM calling
    print("Preparing LSTM data...")
    seq_length = 24
    X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm, scaler = prepare_lstm_data(
        df, seq_length=seq_length
    )

    print("Creating LSTM model...")
    lstm_model = LTSMModel(input_size=1, hidden_size=64, num_layers=2, output_size=1)

    print("Training LSTM...")
    lstm_model = train_lstm(lstm_model, X_train_lstm, y_train_lstm, epochs=50)

    print("Evaluating LSTM...")
    predictions, y_test_original, lstm_rmse = evaluate_lstm(lstm_model, X_test_lstm, y_test_lstm, scaler)
    print(f"   LSTM RMSE: {lstm_rmse:.2f}")

    # Model Comparison
    models = {
        'Linear Regression': lr_results['rmse'],
        'Random Forest': rf_results['rmse'],
        'Tuned XGBoost': xg_results['rmse'],
        'LSTM': lstm_rmse
    }

    print("\n" + "=" * 60)
    print("Final Model Comparison")
    print("=" * 60)
    for name, rmse in models.items():
        print(f"{name}: {rmse:.2f}")

    best = min(models, key=models.get)
    print(f"Best model: {best} with RMSE: {models[best]:.2f}")

    
    # Feature Importance (XGBoost)
    xgb_model = best_xgb_pipeline.named_steps['model']
    feature_importance = xgb_model.feature_importances_
    feature_names = best_xgb_pipeline.named_steps['preprocessor'].get_feature_names_out()

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)

    print(" Top 10 Most Important Features:")
    print("=" * 40)
    for i, row in importance_df.head(10).iterrows():
        print(f"{row['feature']:25} : {row['importance']:.4f}")

    # Save sklearn models
    joblib.dump(best_xgb_pipeline, 'models/xgb_model.pkl')

    # Save LSTM
    torch.save(lstm_model.state_dict(), 'models/lstm_model.pth')

if __name__ == "__main__":
    main()