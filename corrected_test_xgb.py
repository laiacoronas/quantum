import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df, columns_to_drop):
    df_cleaned = df.drop(columns=columns_to_drop)
    return df_cleaned

def train_model(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = XGBRegressor()

    param_grid = {
        'n_estimators': [1000],
        'max_depth': [20],
        'min_child_weight': [1],
        'learning_rate': [0.1],
        'subsample': [0.6]
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

    return re_percentage, re_std

# File paths
train_file_path = r"C:\Users\lclai\Desktop\datasets_corrected\training\pubchem_gse.csv"
test_file_path = r"C:\Users\lclai\Desktop\datasets_corrected\test\pubchem.csv"  

# Columns to drop and target column
columns_to_drop = ['id', 'mf']  
target_column = 'energy'  

# Load and preprocess the training data
train_data = load_data(train_file_path)
y_train = train_data[target_column]
train_cleaned = preprocess_data(train_data, columns_to_drop)

X_train = train_cleaned.drop(columns=[target_column])

# Train the model on the full dataset
model, scaler = train_model(X_train, y_train)

# Load and preprocess the test data
test_data = load_data(test_file_path)
y_test = test_data[target_column]  
test_cleaned = preprocess_data(test_data, columns_to_drop)

X_test = test_cleaned.drop(columns=[target_column])

# Evaluate the model on the new dataset
re_percentage, re_std = evaluate_model(model, scaler, X_test, y_test)

if re_percentage is not None:
    print(f"Relative Error (RE%): {re_percentage:.2f}%")
    print(f"Standard Deviation of RE%: {re_std:.2f}%")
