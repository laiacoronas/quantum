import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df, columns_to_drop):
    df_cleaned = df.drop(columns=columns_to_drop)
    return df_cleaned

def train_evaluate_model(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestRegressor()

    param_grid = {
        'n_estimators': [300,500,1000],
        'max_depth': [5, 10, 20, 30],
        'min_samples_split': [3, 5, 10],
        'max_features':['sqrt']
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

    return re_percentage, re_std

file_path = r"C:\Users\lclai\Desktop\datasets_corrected\training\pubchem_gse.csv"

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