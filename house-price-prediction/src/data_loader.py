"""
Data loader module for house price prediction.

This module handles loading and reading data from various sources.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class DataLoader:
    """Loader class for Kaggle House Prices dataset."""
    
    def load_data(self, train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load training and test datasets from CSV files.
        
        Parameters
        ----------
        train_path : str
            Path to the training CSV file (train.csv).
        test_path : str
            Path to the test CSV file (test.csv).
        
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Tuple containing (train_df, test_df).
        
        Raises
        ------
        FileNotFoundError
            If either CSV file is not found.
        """
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            print(f"Data loaded successfully!")
            print(f"Training set shape: {train_df.shape}")
            print(f"Test set shape: {test_df.shape}")
            
            # Print target statistics
            if "SalePrice" in train_df.columns:
                print(f"\nTarget Variable (SalePrice) Statistics:")
                print(f"  Mean: ${train_df['SalePrice'].mean():,.2f}")
                print(f"  Min:  ${train_df['SalePrice'].min():,.2f}")
                print(f"  Max:  ${train_df['SalePrice'].max():,.2f}")
            
            return train_df, test_df
        
        except FileNotFoundError as e:
            print(f"Error: Could not find the file. {e}")
            raise
    
    def get_feature_info(self, df: pd.DataFrame) -> Dict:
        """
        Get comprehensive information about features in the dataframe.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe to analyze.
        
        Returns
        -------
        Dict
            Dictionary containing:
            - 'total_columns': Total number of columns
            - 'numeric_columns': List of numeric column names
            - 'categorical_columns': List of categorical column names
            - 'missing_values': Dictionary with missing value counts per column
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        missing_values = df.isnull().sum().to_dict()
        
        feature_info = {
            'total_columns': df.shape[1],
            'numeric_columns': numeric_cols,
            'categorical_columns': categorical_cols,
            'missing_values': missing_values
        }
        
        return feature_info
    
    def split_features_target(
        self, 
        df: pd.DataFrame, 
        target_col: str = "SalePrice"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split the dataframe into features (X) and target (y).
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe containing features and target.
        target_col : str, optional
            Name of the target column (default is "SalePrice").
        
        Returns
        -------
        Tuple[pd.DataFrame, pd.Series]
            Tuple containing (X_features, y_target).
        
        Raises
        ------
        KeyError
            If the target column is not found in the dataframe.
        """
        if target_col not in df.columns:
            raise KeyError(f"Target column '{target_col}' not found in dataframe.")
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        print(f"\nFeatures and target split:")
        print(f"  Features shape: {X.shape}")
        print(f"  Target shape: {y.shape}")
        
        return X, y
