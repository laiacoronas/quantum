import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

def load_data(file_path):
    df = pd.read_csv(file_path).fillna(0)  # Fill NA values with 0
    return df

def preprocess_data(df, columns_to_drop):
    df_cleaned = df.drop(columns=columns_to_drop)
    return df_cleaned

def align_train_test_columns(X_train, X_test):
    # Find columns in train but not in test and add them to test with zeros
    for col in X_train.columns:
        if col not in X_test.columns:
            X_test[col] = 0
    # Find columns in test but not in train and add them to train with zeros
    for col in X_test.columns:
        if col not in X_train.columns:
            X_train[col] = 0
    # Ensure same column order
    X_test = X_test[X_train.columns]
    return X_train, X_test

def train_model(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestRegressor()

    param_grid = {
        'n_estimators': [1000],
        'max_depth': [10],
        'min_samples_split': [3],
        'max_features': ["sqrt"]
    }

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)

    print("Best hyperparameters:", grid_search.best_params_)
    
    return grid_search.best_estimator_, scaler

def evaluate_model(model, scaler, X_test, y_test):
    X_test_scaled = scaler.transform(X_test)
    
    predictions = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, predictions)

    # Calculate Relative Error (RE%) and its standard deviation
    mean_y_test = y_test.mean()
    if mean_y_test == 0:
        print("The mean of y_test is zero. Cannot calculate Relative Error.")
        return None, None

    absolute_percentage_errors = np.abs((predictions - y_test) / y_test) * 100
    re_percentage = np.mean(absolute_percentage_errors)
    re_std = np.std(absolute_percentage_errors)
    
        # Plot predictions and ground truth vs index
    plt.figure(figsize=(12, 6))
    sample_indices = range(len(y_test))
    plt.plot(sample_indices, y_test, 'o-', color='blue', label='Ground Truth', markersize=4)
    plt.plot(sample_indices, predictions, 'o-', color='orange', label='Predictions', markersize=4, alpha=0.7)
    
    plt.xlabel("Sample Index")
    plt.ylabel("Energy")
    plt.title("Predictions vs Ground Truth for Energy")
    plt.legend()
    plt.show()


    return re_percentage, re_std

# File paths
train_file_path = r"C:\Users\lclai\Desktop\datasets_corrected\training\combined.csv"
test_file_path = r"C:\Users\lclai\Desktop\datasets_corrected\test\combined.csv"  

# Columns to drop and target column
columns_to_drop = ['id', 'mf']  
target_column = 'energy'  

# Load and preprocess the training data
train_data = load_data(train_file_path)
y_train = train_data[target_column]
train_cleaned = preprocess_data(train_data, columns_to_drop)

X_train = train_cleaned.drop(columns=[target_column])

# Load and preprocess the test data
test_data = load_data(test_file_path)
y_test = test_data[target_column]  
test_cleaned = preprocess_data(test_data, columns_to_drop)

X_test = test_cleaned.drop(columns=[target_column])

# Align columns between train and test sets
X_train, X_test = align_train_test_columns(X_train, X_test)

# Train the model on the full dataset
model, scaler = train_model(X_train, y_train)

# Evaluate the model on the new dataset
re_percentage, re_std = evaluate_model(model, scaler, X_test, y_test)

if re_percentage is not None:
    print(f"Relative Error (RE%): {re_percentage:.2f}%")
    print(f"Standard Deviation of RE%: {re_std:.2f}%")
