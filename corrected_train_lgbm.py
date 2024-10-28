import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df, columns_to_drop):
    df_cleaned = df.drop(columns=columns_to_drop).fillna(0)  # Replace NA with 0
    return df_cleaned

def train_evaluate_model(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LGBMRegressor()

    param_grid = {
        'n_estimators': [500, 1000],
        'max_depth': [20, 30],  
        'min_child_samples': [5, 10],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.6, 1.0]
    }

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)

    print("Best hyperparameters:", grid_search.best_params_)
    
    predictions = grid_search.best_estimator_.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, predictions)

    # Calculate Relative Error (RE%) and its standard deviation
    mean_y_test = y_test.mean()
    if mean_y_test == 0:
        print("The mean of y_test is zero. Cannot calculate Relative Error.")
        return None, None

    # Calculate the absolute percentage errors
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

file_path = r"C:\Users\lclai\Desktop\datasets_corrected\training\combined.csv"

columns_to_drop = ['id', 'mf']  
target_column = 'energy'  

data = load_data(file_path)
y = data[target_column]
data_cleaned = preprocess_data(data, columns_to_drop)

X = data_cleaned.drop(columns=[target_column])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

re_percentage, re_std = train_evaluate_model(X_train, y_train, X_test, y_test)

if re_percentage is not None:
    print(f"Relative Error (RE%): {abs(re_percentage):.2f}%")
    print(f"Standard Deviation of RE%: {re_std:.2f}%")
