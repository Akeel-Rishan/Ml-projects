"""
Data preprocessing module for house price prediction.

This module handles data cleaning, transformation, and feature engineering.
"""

import numpy as np
import pandas as pd
from typing import List
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class Preprocessor:
    """Preprocessor for handling data cleaning and transformation using sklearn pipelines."""
    
    def __init__(self):
        """
        Initialize the Preprocessor with feature lists for the House Prices dataset.
        
        Defines numeric and categorical features from the Kaggle House Prices dataset.
        """
        # Top ~20 most useful numeric features
        self.numeric_features: List[str] = [
            'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
            'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
            '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageYrBlt', 'GarageCars',
            'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'PoolArea', 'MoSold'
        ]
        
        # Top ~20 most useful categorical features
        self.categorical_features: List[str] = [
            'MSZoning', 'LotShape', 'LandContour', 'LotConfig', 'Neighborhood',
            'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'Functional',
            'FireplaceQu', 'PavedDrive', 'RoofMatl', 'Exterior1st', 'Exterior2nd'
        ]
        
        self.pipeline: Pipeline = None
    
    def build_pipeline(self) -> Pipeline:
        """
        Build a sklearn ColumnTransformer pipeline for preprocessing.
        
        Creates a pipeline with:
        - Numeric features: SimpleImputer (median) → StandardScaler
        - Categorical features: SimpleImputer (most_frequent) → OneHotEncoder
        
        Returns
        -------
        Pipeline
            Configured sklearn Pipeline for preprocessing.
        """
        # Numeric pipeline
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical pipeline
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine both transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('numeric', numeric_transformer, self.numeric_features),
                ('categorical', categorical_transformer, self.categorical_features)
            ]
        )
        
        self.pipeline = preprocessor
        return self.pipeline
    
    def fit_transform(self, X_train: pd.DataFrame) -> np.ndarray:
        """
        Fit the pipeline on training data and transform it.
        
        Parameters
        ----------
        X_train : pd.DataFrame
            Training features dataframe.
        
        Returns
        -------
        np.ndarray
            Transformed training features as numpy array.
        """
        if self.pipeline is None:
            self.build_pipeline()
        
        X_transformed = self.pipeline.fit_transform(X_train)
        print(f"Pipeline fitted and training data transformed.")
        print(f"Input shape: {X_train.shape} → Output shape: {X_transformed.shape}")
        
        return X_transformed
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform data using the fitted pipeline.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features dataframe to transform.
        
        Returns
        -------
        np.ndarray
            Transformed features as numpy array.
        
        Raises
        ------
        RuntimeError
            If pipeline has not been fitted yet.
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline must be fitted before transform. Call fit_transform first.")
        
        X_transformed = self.pipeline.transform(X)
        return X_transformed
    
    def get_feature_names(self) -> List[str]:
        """
        Get the names of all features after transformation.
        
        Returns
        -------
        List[str]
            List of transformed feature names including OneHotEncoded categorical features.
        
        Raises
        ------
        RuntimeError
            If pipeline has not been fitted yet.
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline must be fitted before getting feature names. Call fit_transform first.")
        
        feature_names = []
        
        # Get numeric feature names (unchanged after scaling)
        feature_names.extend(self.numeric_features)
        
        # Get categorical feature names from OneHotEncoder
        cat_encoder = self.pipeline.named_transformers_['categorical'].named_steps['onehot']
        cat_feature_names = cat_encoder.get_feature_names_out(self.categorical_features).tolist()
        feature_names.extend(cat_feature_names)
        
        return feature_names
