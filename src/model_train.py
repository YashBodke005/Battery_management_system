"""
Model training module for battery health monitoring.
Includes functions for training different regression models.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
import joblib
import os
import time

def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None, random_state=42):
    """
    Train a Random Forest Regression model.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        n_estimators (int): Number of trees
        max_depth (int): Maximum depth of trees
        random_state (int): Random seed for reproducibility
    
    Returns:
        RandomForestRegressor: Trained model
    """
    print("Training Random Forest Regression model...")
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"Random Forest model trained in {training_time:.2f} seconds")
    return model

def train_decision_tree(X_train, y_train, max_depth=None, random_state=42):
    """
    Train a Decision Tree Regression model.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        max_depth (int): Maximum depth of the tree
        random_state (int): Random seed for reproducibility
    
    Returns:
        DecisionTreeRegressor: Trained model
    """
    print("Training Decision Tree Regression model...")
    
    model = DecisionTreeRegressor(
        max_depth=max_depth,
        random_state=random_state
    )
    
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"Decision Tree model trained in {training_time:.2f} seconds")
    return model

def train_svr(X_train, y_train, kernel='rbf', C=1.0, epsilon=0.1):
    """
    Train a Support Vector Regression model.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        kernel (str): Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
        C (float): Regularization parameter
        epsilon (float): Epsilon in the epsilon-SVR model
    
    Returns:
        SVR: Trained model
    """
    print("Training Support Vector Regression model...")
    
    model = SVR(kernel=kernel, C=C, epsilon=epsilon)
    
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"SVR model trained in {training_time:.2f} seconds")
    return model

def train_linear_regression(X_train, y_train):
    """
    Train a Linear Regression model.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
    
    Returns:
        LinearRegression: Trained model
    """
    print("Training Linear Regression model...")
    
    model = LinearRegression()
    
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"Linear Regression model trained in {training_time:.2f} seconds")
    return model

def train_knn(X_train, y_train, n_neighbors=5, weights='uniform'):
    """
    Train a K-Nearest Neighbor Regression model.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        n_neighbors (int): Number of neighbors
        weights (str): Weight function ('uniform', 'distance')
    
    Returns:
        KNeighborsRegressor: Trained model
    """
    print("Training K-Nearest Neighbor Regression model...")
    
    model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)
    
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"KNN model trained in {training_time:.2f} seconds")
    return model

def optimize_random_forest(X_train, y_train, X_test, y_test, cv=3, random_state=42):
    """
    Optimize Random Forest Regression model using GridSearchCV.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        X_test (pd.DataFrame): Testing features
        y_test (pd.Series): Testing target
        cv (int): Number of cross-validation folds
        random_state (int): Random seed for reproducibility
    
    Returns:
        RandomForestRegressor: Optimized model
    """
    print("Optimizing Random Forest Regression model using GridSearchCV...")
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    base_model = RandomForestRegressor(random_state=random_state)
    
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        n_jobs=-1,
        scoring='neg_mean_squared_error',
        verbose=1
    )
    
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"Random Forest optimization completed in {training_time:.2f} seconds")
    print(f"Best parameters: {grid_search.best_params_}")
    
    best_model = grid_search.best_estimator_
    return best_model, grid_search.best_params_

def train_all_models(X_train, y_train):
    """
    Train all regression models.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
    
    Returns:
        dict: Dictionary of trained models
    """
    models = {
        'random_forest': train_random_forest(X_train, y_train),
        'decision_tree': train_decision_tree(X_train, y_train),
        'svr': train_svr(X_train, y_train),
        'linear_regression': train_linear_regression(X_train, y_train),
        'knn': train_knn(X_train, y_train)
    }
    
    return models

def save_model(model, model_name, models_dir='models'):
    """
    Save a trained model to disk.
    
    Args:
        model: Trained model
        model_name (str): Name of the model
        models_dir (str): Directory to save the model
    
    Returns:
        str: Path to the saved model
    """
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, f"{model_name}.joblib")
    joblib.dump(model, model_path)
    
    print(f"Model saved to {model_path}")
    return model_path

def save_all_models(models, models_dir='models'):
    """
    Save all trained models to disk.
    
    Args:
        models (dict): Dictionary of trained models
        models_dir (str): Directory to save the models
    
    Returns:
        dict: Dictionary of paths to saved models
    """
    model_paths = {}
    
    for model_name, model in models.items():
        model_path = save_model(model, model_name, models_dir)
        model_paths[model_name] = model_path
    
    return model_paths

def load_model(model_path):
    """
    Load a trained model from disk.
    
    Args:
        model_path (str): Path to the saved model
    
    Returns:
        object: Loaded model
    """
    try:
        model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None