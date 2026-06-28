from src.data_loader import load_data, prepare_data
from src.preprocessing import split_time_series
from src.model import (
    create_linear_regression_pipeline,
    train_model,
    evaluate_model
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
    
    # 3. Create and train pipeline
    print("\n🤖 Training Linear Regression...")
    pipeline = create_linear_regression_pipeline()
    pipeline = train_model(pipeline, X_train, y_train)
    
    # 4. Evaluate
    print("\n📈 Evaluating...")
    results = evaluate_model(pipeline, X_test, y_test)
    
    # 5. Print results
    print("\n" + "-" * 40)
    print("RESULTS")
    print("-" * 40)
    print(f"   RMSE: {results['rmse']:.2f}")
    print(f"   MAE:  {results['mae']:.2f}")
    print(f"   R²:   {results['r2']:.4f}")
    print("-" * 40)

if __name__ == "__main__":
    main()

    
