"""
Model evaluation module for house price prediction.

This module contains the Evaluator class for evaluating and comparing model performance.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class Evaluator:
    """Evaluator class for model evaluation, comparison, and visualization."""
    
    def evaluate(
        self, 
        y_true: pd.Series, 
        y_pred: np.ndarray, 
        model_name: str
    ) -> Dict[str, float]:
        """
        Evaluate model performance using regression metrics.
        
        Computes Mean Absolute Error (MAE), Root Mean Squared Error (RMSE),
        and R² Score. Prints a formatted table of results.
        
        Parameters
        ----------
        y_true : pd.Series
            True target values.
        y_pred : np.ndarray
            Predicted target values.
        model_name : str
            Name of the model being evaluated.
        
        Returns
        -------
        Dict[str, float]
            Dictionary containing:
            - 'MAE': Mean Absolute Error
            - 'RMSE': Root Mean Squared Error
            - 'R2': R² Score
        """
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2
        }
        
        # Print formatted results table
        print(f"\n{'='*50}")
        print(f"Evaluation Results: {model_name}")
        print(f"{'='*50}")
        print(f"{'Metric':<20} {'Value':<20}")
        print(f"{'-'*50}")
        print(f"{'MAE':<20} ${mae:,.2f}")
        print(f"{'RMSE':<20} ${rmse:,.2f}")
        print(f"{'R² Score':<20} {r2:.4f}")
        print(f"{'='*50}\n")
        
        return metrics
    
    def compare_models(self, results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Compare multiple models based on their evaluation metrics.
        
        Parameters
        ----------
        results : Dict[str, Dict[str, float]]
            Dictionary mapping model names to their metrics dictionaries.
            Example: {
                'linear_regression': {'MAE': 20000, 'RMSE': 25000, 'R2': 0.85},
                'random_forest': {'MAE': 15000, 'RMSE': 20000, 'R2': 0.90}
            }
        
        Returns
        -------
        pd.DataFrame
            DataFrame with models as rows and metrics as columns,
            sorted by RMSE in ascending order.
        """
        comparison_df = pd.DataFrame(results).T
        comparison_df = comparison_df.sort_values('RMSE', ascending=True)
        
        print(f"\n{'='*60}")
        print("Model Comparison (sorted by RMSE)")
        print(f"{'='*60}")
        print(comparison_df.to_string())
        print(f"{'='*60}\n")
        
        return comparison_df
    
    def plot_predictions(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        model_name: str,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot actual vs predicted values with a perfect prediction diagonal.
        
        Parameters
        ----------
        y_true : pd.Series
            True target values.
        y_pred : np.ndarray
            Predicted target values.
        model_name : str
            Name of the model for the plot title.
        save_path : Optional[str]
            File path to save the plot. If None, plot is displayed but not saved.
        """
        plt.figure(figsize=(10, 6))
        
        # Scatter plot of actual vs predicted
        plt.scatter(y_true, y_pred, alpha=0.5, s=50, edgecolors='k', linewidth=0.5)
        
        # Perfect prediction diagonal line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        plt.xlabel('Actual Price ($)', fontsize=12)
        plt.ylabel('Predicted Price ($)', fontsize=12)
        plt.title(f"{model_name} — Actual vs Predicted", fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_residuals(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        model_name: str,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot residuals (predicted vs residuals) with a horizontal line at 0.
        
        Parameters
        ----------
        y_true : pd.Series
            True target values.
        y_pred : np.ndarray
            Predicted target values.
        model_name : str
            Name of the model for the plot title.
        save_path : Optional[str]
            File path to save the plot. If None, plot is displayed but not saved.
        """
        residuals = y_true - y_pred
        
        plt.figure(figsize=(10, 6))
        
        # Scatter plot of predictions vs residuals
        plt.scatter(y_pred, residuals, alpha=0.5, s=50, edgecolors='k', linewidth=0.5)
        
        # Horizontal line at 0
        plt.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero Residual')
        
        plt.xlabel('Predicted Price ($)', fontsize=12)
        plt.ylabel('Residuals ($)', fontsize=12)
        plt.title(f"{model_name} — Residual Plot", fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
