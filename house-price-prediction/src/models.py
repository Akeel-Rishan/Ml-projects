"""
Model training module for house price prediction.

This module contains functions for training various machine learning models.
"""


def train_linear_regression(X_train, y_train):
    """
    Train a Linear Regression model.
    
    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
        Training features.
    y_train : pd.Series or np.ndarray
        Training target values.
    
    Returns
    -------
    model
        Trained Linear Regression model.
    """
    pass


def train_random_forest(X_train, y_train, n_estimators=100):
    """
    Train a Random Forest model.
    
    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
        Training features.
    y_train : pd.Series or np.ndarray
        Training target values.
    n_estimators : int, optional
        Number of trees in the forest (default is 100).
    
    Returns
    -------
    model
        Trained Random Forest model.
    """
    pass


def train_gradient_boosting(X_train, y_train, n_estimators=100):
    """
    Train a Gradient Boosting model.
    
    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
        Training features.
    y_train : pd.Series or np.ndarray
        Training target values.
    n_estimators : int, optional
        Number of boosting stages (default is 100).
    
    Returns
    -------
    model
        Trained Gradient Boosting model.
    """
    pass
