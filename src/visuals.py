"""
Visualization module for battery health monitoring.
Includes functions for data visualization and model evaluation plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_output_dir(output_dir='results'):
    """
    Create output directory if it doesn't exist.
    
    Args:
        output_dir (str): Directory to create
    """
    os.makedirs(output_dir, exist_ok=True)

def plot_correlation_heatmap(data, output_dir='results'):
    """
    Plot correlation heatmap of dataset features.
    
    Args:
        data (pd.DataFrame): Input data
        output_dir (str): Directory to save the plot
    
    Returns:
        str: Path to the saved plot
    """
    print("Plotting correlation heatmap...")
    
    plt.figure(figsize=(12, 10))
    correlation = data.corr()
    mask = np.triu(correlation)
    
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, fmt='.2f', mask=mask)
    plt.title('Feature Correlation Heatmap', fontsize=16)
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    create_output_dir(output_dir)
    
    # Save the plot
    output_path = os.path.join(output_dir, 'correlation_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Correlation heatmap saved to {output_path}")
    return output_path

def plot_feature_importance(model, feature_names, model_name=None, output_dir='results'):
    """
    Plot feature importance for tree-based models.
    
    Args:
        model: Trained model (must have feature_importances_ attribute)
        feature_names (list): Names of the features
        model_name (str): Name of the model (for display)
        output_dir (str): Directory to save the plot
    
    Returns:
        str: Path to the saved plot
    """
    if not hasattr(model, 'feature_importances_'):
        print("Model does not have feature_importances_ attribute. Skipping feature importance plot.")
        return None
    
    print("Plotting feature importance...")
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Sort importances and feature names
    indices = np.argsort(importances)[::-1]
    sorted_importances = importances[indices]
    sorted_features = [feature_names[i] for i in indices]
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.bar(range(len(sorted_importances)), sorted_importances, align='center')
    plt.xticks(range(len(sorted_importances)), sorted_features, rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    
    if model_name:
        plt.title(f'Feature Importance ({model_name})')
    else:
        plt.title('Feature Importance')
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    create_output_dir(output_dir)
    
    # Save the plot
    output_path = os.path.join(output_dir, f'feature_importance_{model_name}.png' if model_name else 'feature_importance.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Feature importance plot saved to {output_path}")
    return output_path

def plot_actual_vs_predicted(y_true, y_pred, model_name=None, output_dir='results'):
    """
    Plot actual vs predicted values.
    
    Args:
        y_true (array-like): True target values
        y_pred (array-like): Predicted target values
        model_name (str): Name of the model (for display)
        output_dir (str): Directory to save the plot
    
    Returns:
        str: Path to the saved plot
    """
    print("Plotting actual vs predicted values...")
    
    plt.figure(figsize=(10, 8))
    
    # Plot diagonal line (perfect predictions)
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
    
    # Plot actual vs predicted
    plt.scatter(y_true, y_pred, alpha=0.6)
    
    plt.xlabel('Actual RUL')
    plt.ylabel('Predicted RUL')
    
    if model_name:
        plt.title(f'Actual vs Predicted RUL ({model_name})')
    else:
        plt.title('Actual vs Predicted RUL')
    
    plt.grid(True)
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    create_output_dir(output_dir)
    
    # Save the plot
    output_path = os.path.join(output_dir, f'actual_vs_predicted_{model_name}.png' if model_name else 'actual_vs_predicted.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Actual vs predicted plot saved to {output_path}")
    return output_path

def plot_residuals(y_true, y_pred, model_name=None, output_dir='results'):
    """
    Plot residuals (prediction errors).
    
    Args:
        y_true (array-like): True target values
        y_pred (array-like): Predicted target values
        model_name (str): Name of the model (for display)
        output_dir (str): Directory to save the plot
    
    Returns:
        str: Path to the saved plot
    """
    print("Plotting residuals...")
    
    residuals = y_true - y_pred
    
    plt.figure(figsize=(12, 10))
    
    # Residuals vs Predicted
    plt.subplot(2, 1, 1)
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='k', linestyle='--', lw=2)
    plt.xlabel('Predicted RUL')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted Values')
    plt.grid(True)
    
    # Residuals distribution
    plt.subplot(2, 1, 2)
    sns.histplot(residuals, kde=True)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Distribution of Residuals')
    plt.grid(True)
    
    plt.tight_layout()
# Create output directory if it doesn't exist
    create_output_dir(output_dir)
    
    # Save the plot
    output_path = os.path.join(output_dir, f'residuals_{model_name}.png' if model_name else 'residuals.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Residuals plot saved to {output_path}")
    return output_path

def plot_model_comparison(results_df, metric='R2', output_dir='results'):
    """
    Plot model comparison bar chart.
    
    Args:
        results_df (pd.DataFrame): DataFrame of evaluation metrics
        metric (str): Metric to plot
        output_dir (str): Directory to save the plot
    
    Returns:
        str: Path to the saved plot
    """
    print(f"Plotting model comparison for {metric}...")
    
    plt.figure(figsize=(12, 8))
    
    # Sort models by metric value
    sorted_df = results_df.sort_values(by=metric, ascending=False)
    
    # Plot
    ax = sns.barplot(x='Model', y=metric, data=sorted_df)
    
    # Add value labels on top of bars
    for i, v in enumerate(sorted_df[metric]):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    plt.title(f'Model Comparison ({metric})')
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    create_output_dir(output_dir)
    
    # Save the plot
    output_path = os.path.join(output_dir, f'model_comparison_{metric}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Model comparison plot saved to {output_path}")
    return output_path

def plot_learning_curve(model, X_train, y_train, model_name=None, cv=5, output_dir='results'):
    """
    Plot learning curve for a model.
    
    Args:
        model: Trained model
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        model_name (str): Name of the model (for display)
        cv (int): Number of cross-validation folds
        output_dir (str): Directory to save the plot
    
    Returns:
        str: Path to the saved plot
    """
    from sklearn.model_selection import learning_curve
    
    print("Plotting learning curve...")
    
    # Calculate learning curve
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=cv, scoring='r2',
        train_sizes=np.linspace(0.1, 1.0, 10)
    )
    
    # Calculate mean and std for train and test scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
    plt.plot(train_sizes, test_mean, 'o-', color='g', label='Cross-validation score')
    
    # Plot error bands
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('R² Score')
    
    if model_name:
        plt.title(f'Learning Curve ({model_name})')
    else:
        plt.title('Learning Curve')
    
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    create_output_dir(output_dir)
    
    # Save the plot
    output_path = os.path.join(output_dir, f'learning_curve_{model_name}.png' if model_name else 'learning_curve.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Learning curve plot saved to {output_path}")
    return output_path

def plot_pair_grid(data, target_column='RUL', sample_n=1000, output_dir='results'):
    """
    Plot pair grid of features and target.
    
    Args:
        data (pd.DataFrame): Input data
        target_column (str): Name of the target column
        sample_n (int): Number of samples to use for plotting
        output_dir (str): Directory to save the plot
    
    Returns:
        str: Path to the saved plot
    """
    print("Plotting feature pair grid...")
    
    # Sample data if necessary
    if len(data) > sample_n:
        data_sample = data.sample(n=sample_n, random_state=42)
    else:
        data_sample = data
    
    # Select columns to plot (avoid plotting too many features)
    corr_with_target = data.corr()[target_column].abs().sort_values(ascending=False)
    top_features = list(corr_with_target.index[:5])  # Top 5 features + target
    if target_column not in top_features:
        top_features.append(target_column)
    
    data_to_plot = data_sample[top_features]
    
    # Plot
    plt.figure(figsize=(15, 12))
    grid = sns.PairGrid(data_to_plot, diag_sharey=False, corner=True)
    grid.map_lower(sns.scatterplot, alpha=0.5)
    grid.map_diag(sns.histplot, kde=True)
    
    plt.suptitle('Feature Pair Grid', y=1.02, fontsize=16)
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    create_output_dir(output_dir)
    
    # Save the plot
    output_path = os.path.join(output_dir, 'pair_grid.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Pair grid plot saved to {output_path}")
    return output_path

def create_dashboard(results_df, predictions_dict, y_test, output_dir='results'):
    """
    Create a comprehensive dashboard of model results.
    
    Args:
        results_df (pd.DataFrame): DataFrame of evaluation metrics
        predictions_dict (dict): Dictionary of predictions for each model
        y_test (pd.Series): Testing target values
        output_dir (str): Directory to save the dashboard
    
    Returns:
        str: Path to the saved dashboard
    """
    print("Creating model evaluation dashboard...")
    
    # Create a figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Model comparison bar chart for R²
    plt.subplot(2, 2, 1)
    sorted_df = results_df.sort_values(by='R2', ascending=False)
    ax1 = sns.barplot(x='Model', y='R2', data=sorted_df)
    for i, v in enumerate(sorted_df['R2']):
        ax1.text(i, v + 0.01, f'{v:.3f}', ha='center')
    plt.title('Model Comparison (R²)')
    plt.ylabel('R² Score')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    
    # 2. Model comparison bar chart for RMSE
    plt.subplot(2, 2, 2)
    sorted_df = results_df.sort_values(by='RMSE', ascending=True)
    ax2 = sns.barplot(x='Model', y='RMSE', data=sorted_df)
    for i, v in enumerate(sorted_df['RMSE']):
        ax2.text(i, v + 0.5, f'{v:.2f}', ha='center')
    plt.title('Model Comparison (RMSE)')
    plt.ylabel('RMSE')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    
    # 3. Actual vs Predicted scatter plot for the best model
    plt.subplot(2, 2, 3)
    best_model = results_df.loc[results_df['R2'].idxmax(), 'Model']
    best_predictions = predictions_dict[best_model]
    
    # Plot diagonal line (perfect predictions)
    min_val = min(np.min(y_test), np.min(best_predictions))
    max_val = max(np.max(y_test), np.max(best_predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
    
    # Plot actual vs predicted
    plt.scatter(y_test, best_predictions, alpha=0.6)
    plt.xlabel('Actual RUL')
    plt.ylabel('Predicted RUL')
    plt.title(f'Actual vs Predicted RUL (Best Model: {best_model})')
    plt.grid(True)
    
    # 4. Residuals histogram for the best model
    plt.subplot(2, 2, 4)
    residuals = y_test - best_predictions
    sns.histplot(residuals, kde=True)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Residuals (Best Model: {best_model})')
    plt.grid(True)
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    create_output_dir(output_dir)
    
    # Save the dashboard
    output_path = os.path.join(output_dir, 'model_evaluation_dashboard.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Model evaluation dashboard saved to {output_path}")
    return output_path