"""
Main script for battery health monitoring using machine learning.
This script orchestrates the entire process from data loading to model evaluation.
"""

import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Optionally, set the plotting style
# plt.style.use('seaborn-v0_8-darkgrid')
# sns.set(font_scale=1.2)

# Import custom modules.
# If your modules are located in a subdirectory (e.g., "src"), modify the import statements accordingly.
from src.preprocess import preprocess_pipeline
from src.model_train import train_all_models, save_all_models, optimize_random_forest
from src.model_evaluation import evaluate_all_models, save_results, find_best_model
from src.visuals import (
    plot_correlation_heatmap, plot_feature_importance,
    plot_actual_vs_predicted, plot_residuals, plot_model_comparison, plot_pair_grid
)

def main():
    # Start timing
    start_time = time.time()
    
    # Configuration
    DATA_PATH = 'data/Book1.csv'
    TARGET_COLUMN = 'RUL'
    TEST_SIZE = 0.25
    RANDOM_STATE = 42
    OPTIMIZE_BEST_MODEL = True
    
    print("=" * 80)
    print("Battery Health Monitoring System")
    print("=" * 80)
    
    # ------------------------#
    # Step 1: Data Preprocessing
    # ------------------------#
    print("\nStep 1: Data Preprocessing")
    print("-" * 50)
    
    # Preprocess the data (load, handle missing values/outliers, split and normalize)
    preprocess_out = preprocess_pipeline(DATA_PATH, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE)
    if preprocess_out is None:
        print("Preprocessing failed. Check the data file and retry.")
        return
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names = preprocess_out
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Scaler saved to models/scaler.pkl")
    # Create a combined training DataFrame for visualization if needed
    train_data = pd.DataFrame(X_train_scaled, columns=feature_names)
    train_data[TARGET_COLUMN] = y_train.values
    
    # ------------------------#
    # Step 2: Data Visualization
    # ------------------------#
    print("\nStep 2: Data Visualization")
    print("-" * 50)
    
    # Load the original data for visualization purposes
    original_data = pd.read_csv(DATA_PATH)
    
    # Plot correlation heatmap of the original dataset
    plot_correlation_heatmap(original_data)
    
    # Plot a pair grid of features and target (replacing functions like plot_pairplot or plot_feature_distribution)
    plot_pair_grid(original_data, target_column=TARGET_COLUMN)
    
    # ------------------------#
    # Step 3: Model Training
    # ------------------------#
    print("\nStep 3: Model Training")
    print("-" * 50)
    
    # Train all regression models
    models = train_all_models(X_train_scaled, y_train)
    
    # Save all the trained models to disk
    model_paths = save_all_models(models)
    
    # ------------------------#
    # Step 4: Model Evaluation
    # ------------------------#
    print("\nStep 4: Model Evaluation")
    print("-" * 50)
    
    # Evaluate the models on the test set
    results_df, predictions = evaluate_all_models(models, X_test_scaled, y_test)
    
    # Save the evaluation results to a CSV file
    save_results(results_df)
    
    # ------------------------#
    # Step 5: Result Visualization
    # ------------------------#
    print("\nStep 5: Result Visualization")
    print("-" * 50)
    
    # Plot actual vs predicted values and residuals for each model
    for model_name, y_pred in predictions.items():
        plot_actual_vs_predicted(y_test, y_pred, model_name)
        plot_residuals(y_test, y_pred, model_name)
    
    # Plot a model comparison bar chart based on the R² metric (you could change the metric if desired)
    plot_model_comparison(results_df, metric='R2')
    
    # Plot feature importance for tree-based models (if available)
    for model_name, model in models.items():
        if hasattr(model, 'feature_importances_'):
            plot_feature_importance(model, feature_names, model_name)
    
    # ------------------------#
    # Step 6: Model Optimization
    # ------------------------#
    print("\nStep 6: Model Optimization")
    print("-" * 50)
    
    # Identify the best model based on R² and RMSE
    best_model_name = find_best_model(results_df, metric='R2', higher_is_better=True)
    best_rmse_model_name = find_best_model(results_df, metric='RMSE', higher_is_better=False)
    print(f"Best model based on R²: {best_model_name}")
    print(f"Best model based on RMSE: {best_rmse_model_name}")
    
    # If the Random Forest model is the best (or among the best by RMSE), optimize it
    if OPTIMIZE_BEST_MODEL and (best_model_name == 'random_forest' or best_rmse_model_name == 'random_forest'):
        print("\nOptimizing Random Forest model...")
        
        # Optimize Random Forest using GridSearchCV
        optimized_rf, best_params = optimize_random_forest(
            X_train_scaled, y_train, X_test_scaled, y_test, cv=3, random_state=RANDOM_STATE
        )
        print("\nBest parameters:", best_params)
        
        # Evaluate the optimized Random Forest model
        optimized_results, optimized_pred = evaluate_all_models(
            {'optimized_random_forest': optimized_rf}, X_test_scaled, y_test
        )
        
        # Plot results for the optimized model
        plot_actual_vs_predicted(y_test, optimized_pred['optimized_random_forest'], 'optimized_random_forest')
        plot_residuals(y_test, optimized_pred['optimized_random_forest'], 'optimized_random_forest')
        plot_feature_importance(optimized_rf, feature_names, 'optimized_random_forest')
        
        # Save the optimized model
        save_all_models({'optimized_random_forest': optimized_rf})
    
    # ------------------------#
    # End of Pipeline
    # ------------------------#
    end_time = time.time()
    execution_time = end_time - start_time
    print("\n" + "=" * 80)
    print(f"Execution completed in {execution_time:.2f} seconds")
    print("=" * 80)

if __name__ == "__main__":
    main()