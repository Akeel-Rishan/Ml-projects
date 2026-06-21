"""
Model training module for house price prediction.

This module contains the ModelTrainer class for training various machine learning models.
"""

import time
import joblib
import numpy as np
import pandas as pd
from typing import Any, Dict
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


class ModelTrainer:
    """Trainer class for managing model training, prediction, and persistence."""
    
    def __init__(self):
        """
        Initialize the ModelTrainer with available models.
        
        Sets up two regression models:
        - LinearRegression: Simple linear regression model
        - RandomForestRegressor: Ensemble method with 100 trees
        """
        self.models: Dict[str, Any] = {
            "linear_regression": LinearRegression(),
            "random_forest": RandomForestRegressor(n_estimators=100, random_state=42)
        }
        self.training_times: Dict[str, float] = {}
    
    def train(self, model_name: str, X_train: np.ndarray, y_train: pd.Series) -> None:
        """
        Train a specified model on the training data.
        
        Parameters
        ----------
        model_name : str
            Name of the model to train. Must be a key in self.models.
        X_train : np.ndarray
            Training features array.
        y_train : pd.Series
            Training target values.
        
        Raises
        ------
        KeyError
            If model_name is not found in available models.
        """
        if model_name not in self.models:
            raise KeyError(f"Model '{model_name}' not found. Available models: {list(self.models.keys())}")
        
        start_time = time.time()
        
        self.models[model_name].fit(X_train, y_train)
        
        elapsed_time = time.time() - start_time
        self.training_times[model_name] = elapsed_time
        
        print(f"Model '{model_name}' trained successfully in {elapsed_time:.4f} seconds.")
    
    def predict(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions using a trained model.
        
        Parameters
        ----------
        model_name : str
            Name of the model to use for prediction.
        X : np.ndarray
            Features array for prediction.
        
        Returns
        -------
        np.ndarray
            Predicted values.
        
        Raises
        ------
        KeyError
            If model_name is not found in available models.
        """
        if model_name not in self.models:
            raise KeyError(f"Model '{model_name}' not found. Available models: {list(self.models.keys())}")
        
        predictions = self.models[model_name].predict(X)
        return predictions
    
    def save_model(self, model_name: str, path: str) -> None:
        """
        Save a trained model to disk using joblib.
        
        Parameters
        ----------
        model_name : str
            Name of the model to save.
        path : str
            File path where the model will be saved.
        
        Raises
        ------
        KeyError
            If model_name is not found in available models.
        """
        if model_name not in self.models:
            raise KeyError(f"Model '{model_name}' not found. Available models: {list(self.models.keys())}")
        
        joblib.dump(self.models[model_name], path)
        print(f"Model '{model_name}' saved to {path}")
    
    def load_model(self, model_name: str, path: str) -> None:
        """
        Load a model from disk using joblib.
        
        Parameters
        ----------
        model_name : str
            Name to assign to the loaded model.
        path : str
            File path from where the model will be loaded.
        
        Raises
        ------
        FileNotFoundError
            If the model file does not exist.
        """
        try:
            self.models[model_name] = joblib.load(path)
            print(f"Model '{model_name}' loaded from {path}")
        except FileNotFoundError as e:
            print(f"Error: Model file not found at {path}")
            raise
    
    def get_model(self, model_name: str) -> Any:
        """
        Retrieve a model object by name.
        
        Parameters
        ----------
        model_name : str
            Name of the model to retrieve.
        
        Returns
        -------
        Any
            The model object.
        
        Raises
        ------
        KeyError
            If model_name is not found in available models.
        """
        if model_name not in self.models:
            raise KeyError(f"Model '{model_name}' not found. Available models: {list(self.models.keys())}")
        
        return self.models[model_name]
