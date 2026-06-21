"""
Main entry point for the house price prediction project.

This script orchestrates the entire ML pipeline including data loading,
preprocessing, model training, and evaluation.
"""

import os
from sklearn.model_selection import train_test_split
from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.models import ModelTrainer
from src.evaluator import Evaluator


def run_pipeline():
    """Run the complete house price prediction pipeline."""
    
    print("\n" + "="*70)
    print("HOUSE PRICE PREDICTION - ML PIPELINE")
    print("="*70 + "\n")
    
    # ============================================================================
    # Step 1: Load Data
    # ============================================================================
    print("="*70)
    print("Step 1: Loading Data")
    print("="*70)
    
    loader = DataLoader()
    train_df, test_df = loader.load_data("data/train.csv", "data/test.csv")
    
    # ============================================================================
    # Step 2: Split Features and Target
    # ============================================================================
    print("\n" + "="*70)
    print("Step 2: Splitting Features and Target")
    print("="*70 + "\n")
    
    X, y = loader.split_features_target(train_df, target_col="SalePrice")
    
    # ============================================================================
    # Step 3: Train/Validation Split
    # ============================================================================
    print("\n" + "="*70)
    print("Step 3: Train/Validation Split (80/20)")
    print("="*70 + "\n")
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    
    # ============================================================================
    # Step 4: Data Preprocessing
    # ============================================================================
    print("\n" + "="*70)
    print("Step 4: Data Preprocessing")
    print("="*70 + "\n")
    
    preprocessor = Preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_val_transformed = preprocessor.transform(X_val)
    
    feature_names = preprocessor.get_feature_names()
    print(f"Total transformed features: {len(feature_names)}")
    
    # ============================================================================
    # Step 5: Train Models
    # ============================================================================
    print("\n" + "="*70)
    print("Step 5: Training Models")
    print("="*70)
    
    trainer = ModelTrainer()
    
    # Train Linear Regression
    print("\n--- Training Linear Regression ---")
    trainer.train("linear_regression", X_train_transformed, y_train)
    
    # Train Random Forest
    print("\n--- Training Random Forest ---")
    trainer.train("random_forest", X_train_transformed, y_train)
    
    # ============================================================================
    # Step 6: Make Predictions
    # ============================================================================
    print("\n" + "="*70)
    print("Step 6: Making Predictions on Validation Set")
    print("="*70 + "\n")
    
    y_pred_lr = trainer.predict("linear_regression", X_val_transformed)
    y_pred_rf = trainer.predict("random_forest", X_val_transformed)
    
    print("Predictions generated for both models.")
    
    # ============================================================================
    # Step 7: Evaluate Models
    # ============================================================================
    print("\n" + "="*70)
    print("Step 7: Evaluating Models")
    print("="*70)
    
    evaluator = Evaluator()
    
    metrics_lr = evaluator.evaluate(y_val, y_pred_lr, "Linear Regression")
    metrics_rf = evaluator.evaluate(y_val, y_pred_rf, "Random Forest")
    
    # ============================================================================
    # Step 8: Compare Models
    # ============================================================================
    print("\n" + "="*70)
    print("Step 8: Model Comparison")
    print("="*70)
    
    results = {
        "linear_regression": metrics_lr,
        "random_forest": metrics_rf
    }
    
    comparison_df = evaluator.compare_models(results)
    
    # ============================================================================
    # Step 9: Generate and Save Plots
    # ============================================================================
    print("\n" + "="*70)
    print("Step 9: Generating and Saving Plots")
    print("="*70 + "\n")
    
    os.makedirs("outputs", exist_ok=True)
    
    # Linear Regression plots
    print("--- Linear Regression Plots ---")
    evaluator.plot_predictions(
        y_val, y_pred_lr, "Linear Regression",
        save_path="outputs/linear_regression_predictions.png"
    )
    evaluator.plot_residuals(
        y_val, y_pred_lr, "Linear Regression",
        save_path="outputs/linear_regression_residuals.png"
    )
    
    # Random Forest plots
    print("\n--- Random Forest Plots ---")
    evaluator.plot_predictions(
        y_val, y_pred_rf, "Random Forest",
        save_path="outputs/random_forest_predictions.png"
    )
    evaluator.plot_residuals(
        y_val, y_pred_rf, "Random Forest",
        save_path="outputs/random_forest_residuals.png"
    )
    
    # ============================================================================
    # Step 10: Save Trained Models
    # ============================================================================
    print("\n" + "="*70)
    print("Step 10: Saving Trained Models")
    print("="*70 + "\n")
    
    trainer.save_model("linear_regression", "outputs/linear_regression.joblib")
    trainer.save_model("random_forest", "outputs/random_forest.joblib")
    
    # ============================================================================
    # Pipeline Complete
    # ============================================================================
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print("\nOutput Files:")
    print("  - outputs/linear_regression.joblib")
    print("  - outputs/random_forest.joblib")
    print("  - outputs/linear_regression_predictions.png")
    print("  - outputs/linear_regression_residuals.png")
    print("  - outputs/random_forest_predictions.png")
    print("  - outputs/random_forest_residuals.png")
    print("\nBest Model: " + comparison_df.index[0].replace("_", " ").title())
    print("="*70 + "\n")


if __name__ == "__main__":
    run_pipeline()
