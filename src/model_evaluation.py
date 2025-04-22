"""
Model evaluation module for battery health monitoring.
Includes functions for evaluating model performance.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os

def calculate_metrics(y_true, y_pred):
    """
    Calculate regression evaluation metrics.
    
    Args:
        y_true (array-like): True target values
        y_pred (array-like): Predicted target values
    
    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Root Mean Squared Error
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Mean Squared Error
    mse = mean_squared_error(y_true, y_pred)
    
    # Mean Absolute Error
    mae = mean_absolute_error(y_true, y_pred)
    
    # R-squared
    r2 = r2_score(y_true, y_pred)
    
    # Maximum Error
    max_error = np.max(np.abs(y_true - y_pred))
    
    metrics = {
        'R2': r2,
        'RMSE': rmse,
        'MSE': mse,
        'MAE': mae,
        'MAX': max_error
    }
    
    return metrics

def evaluate_model(model, X_test, y_test, model_name=None):
    """
    Evaluate a model on test data.
    
    Args:
        model: Trained model
        X_test (pd.DataFrame): Testing features
        y_test (pd.Series): Testing target
        model_name (str): Name of the model (for display)
    
    Returns:
        dict: Dictionary of evaluation metrics
    """
    if model_name:
        print(f"Evaluating {model_name} model...")
    else:
        print("Evaluating model...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred)
    
    # Print metrics
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.2f}")
    
    return metrics, y_pred

def evaluate_all_models(models, X_test, y_test):
    """
    Evaluate all models on test data.
    
    Args:
        models (dict): Dictionary of trained models
        X_test (pd.DataFrame): Testing features
        y_test (pd.Series): Testing target
    
    Returns:
        pd.DataFrame: DataFrame of evaluation metrics for all models
        dict: Dictionary of predictions for all models
    """
    results = []
    predictions = {}
    
    for model_name, model in models.items():
        metrics, y_pred = evaluate_model(model, X_test, y_test, model_name)
        
        # Add model name to metrics
        metrics['Model'] = model_name
        
        # Add to results list
        results.append(metrics)
        
        # Store predictions
        predictions[model_name] = y_pred
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Reorder columns to put Model first
    cols = ['Model'] + [col for col in results_df.columns if col != 'Model']
    results_df = results_df[cols]
    
    print("\nModel Comparison:")
    print(results_df)
    
    return results_df, predictions

def save_results(results_df, output_dir='results'):
    """
    Save evaluation results to CSV.
    
    Args:
        results_df (pd.DataFrame): DataFrame of evaluation metrics
        output_dir (str): Directory to save the results
    
    Returns:
        str: Path to the saved results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'model_evaluation_results.csv')
    results_df.to_csv(output_path, index=False)
    
    print(f"Evaluation results saved to {output_path}")
    return output_path

def find_best_model(results_df, metric='R2', higher_is_better=True):
    """
    Find the best performing model based on a specific metric.
    
    Args:
        results_df (pd.DataFrame): DataFrame of evaluation metrics
        metric (str): Metric to use for comparison
        higher_is_better (bool): Whether higher values of the metric are better
    
    Returns:
        str: Name of the best model
    """
    if higher_is_better:
        best_idx = results_df[metric].idxmax()
    else:
        best_idx = results_df[metric].idxmin()
    
    best_model = results_df.loc[best_idx, 'Model']
    best_value = results_df.loc[best_idx, metric]
    
    print(f"Best model based on {metric}: {best_model} ({metric}={best_value:.2f})")
    
    return best_model