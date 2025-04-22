"""
Data preprocessing module for battery health monitoring.
Includes functions for loading, cleaning, and preprocessing data.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """
    Load battery data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file
    
    Returns:
        pd.DataFrame: Loaded data
    """
    print(f"Loading data from {file_path}...")
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully. Shape: {data.shape}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def check_missing_values(data):
    """
    Check for missing values in the dataset.
    
    Args:
        data (pd.DataFrame): Input data
    
    Returns:
        pd.Series: Missing value count for each column
    """
    missing_values = data.isnull().sum()
    print("Missing values per column:")
    print(missing_values)
    return missing_values

def handle_missing_values(data, strategy='mean'):
    """
    Handle missing values in the dataset.
    
    Args:
        data (pd.DataFrame): Input data
        strategy (str): Strategy for handling missing values ('mean', 'median', 'drop')
    
    Returns:
        pd.DataFrame: Data with handled missing values
    """
    print(f"Handling missing values using strategy: {strategy}")
    
    if strategy == 'drop':
        data = data.dropna()
    elif strategy == 'mean':
        data = data.fillna(data.mean())
    elif strategy == 'median':
        data = data.fillna(data.median())
    
    print(f"Data shape after handling missing values: {data.shape}")
    return data

def remove_outliers(data, columns=None, z_threshold=3):
    """
    Remove outliers from the dataset using Z-score method.
    
    Args:
        data (pd.DataFrame): Input data
        columns (list): Columns to check for outliers (if None, all numeric columns)
        z_threshold (float): Z-score threshold for outlier detection
    
    Returns:
        pd.DataFrame: Data with outliers removed
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"Removing outliers from {len(columns)} columns using Z-score method...")
    
    #initial_rows = data.shape[0]
    #for column in columns:
    z_scores = np.abs((data[columns] - data[columns].mean()) / data[columns].std())
    #data = data[z_scores < z_threshold]
    mask = (z_scores < z_threshold).all(axis=1)
    cleaned_data = data[mask]

    print(f"Removed {len(data) - len(cleaned_data)} rows as outliers")
    return cleaned_data
    #print(f"Removed {initial_rows - data.shape[0]} rows as outliers")
    #return data

def normalize_data(X_train, X_test):
    """
    Normalize features using StandardScaler.
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Testing features
    
    Returns:
        tuple: (X_train_scaled, X_test_scaled, scaler)
    """
    print("Normalizing data using StandardScaler...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrames to keep column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    return X_train_scaled, X_test_scaled, scaler

def prepare_data(data, target_column='RUL', test_size=0.25, random_state=42):
    """
    Prepare data for model training by splitting into train and test sets.
    
    Args:
        data (pd.DataFrame): Input data
        target_column (str): Name of the target column
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
    
    Returns:
        tuple: (X_train_scaled, X_test_scaled, y_train, y_test, scaler)
    """
    print(f"Preparing data for model training. Target column: {target_column}")
    
    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Data split: X_train {X_train.shape}, X_test {X_test.shape}")
    
    # Normalize features
    X_train_scaled, X_test_scaled, scaler = normalize_data(X_train, X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def preprocess_pipeline(file_path, target_column='RUL', test_size=0.25, random_state=42):
    """
    Complete preprocessing pipeline.
    
    Args:
        file_path (str): Path to the CSV file
        target_column (str): Name of the target column
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
    
    Returns:
        tuple: (X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names)
    """
    # Load data
    data = load_data(file_path)
    if data is None:
        return None
    
    # Check and handle missing values
    check_missing_values(data)
    data = handle_missing_values(data, strategy='mean')
    
    # Remove outliers
    data = remove_outliers(data, z_threshold=3)
    
    # Prepare data for training
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_data(
        data, target_column, test_size, random_state
    )
    
    feature_names = X_train_scaled.columns.tolist()
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names