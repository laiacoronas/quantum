import os, sys
import pandas as pd
import numpy as np
import json
from collections import defaultdict, Counter
import argparse
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
import shap
from lightgbm import LGBMRegressor
import lightgbm as lgb
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


def classifier(X, y, cls, inner_cv, params=None):
    if params is not None:
        if cls == "lgbm":
            return LGBMRegressor(n_estimators=params.get('n_estimators', 100),
                                 learning_rate=params.get('learning_rate', 0.1),
                                 subsample=params.get('subsample', 1.0),
                                 random_state=123), params
        elif cls == "randomf":
            return RandomForestRegressor(random_state=123,
                                         n_estimators=params.get('n_estimators', 100),
                                         max_depth=params.get('max_depth', None)), params
        elif cls == "xgb":
            return XGBRegressor(eval_metric="rmse",
                                random_state=123,
                                n_estimators=params.get('n_estimators', 100),
                                max_depth=params.get('max_depth', 3),
                                learning_rate=params.get('learning_rate', 0.1),
                                subsample=params.get('subsample', 1.0)), params
    else:
        if cls == 'lgbm':
            model = LGBMRegressor(random_state=123)
            param_distributions = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, 30],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0]
            }
        elif cls == 'randomf':
            model = RandomForestRegressor(random_state=123)
            param_distributions = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, 30]
            }
        elif cls == 'xgb':
            model = XGBRegressor(eval_metric="rmse", random_state=123, n_jobs=-1)
            param_distributions = {
                'n_estimators': [50, 75, 100],
                'max_depth': [3, 5],
                'learning_rate': [0.1, 0.2],
                'subsample': [0.8, 1.0]
            }
        
        rs = RandomizedSearchCV(model, param_distributions=param_distributions, n_iter=27, scoring='neg_mean_squared_error', cv=inner_cv, verbose=3)
        rs.fit(X, y.values.ravel())
        return rs.best_estimator_, rs.best_params_


def process_most_common_hyperparams(hyperparam_counts):
    most_common_hyperparams_combined = dict(Counter(hyperparam_counts).most_common())
    processed_hyperparams = {}
    
    for combined_name, count in most_common_hyperparams_combined.items():
        name, value = combined_name.rsplit('_', 1)
        if name in ['n_estimators', 'max_depth']:
            try:
                processed_hyperparams[name] = int(value)
            except ValueError:
                pass
        else:
            try:
                processed_hyperparams[name] = float(value)
            except ValueError:
                processed_hyperparams[name] = value
    
    return processed_hyperparams

def nested_cv(X, y, outer_cv, inner_cv, cls, sel):
    inner_results = []
    outer_results = []

    outer_fold_counter = 1  

    for train_idx_outer, val_idx_outer in outer_cv.split(X, y):
        feature_counts = defaultdict(int)
        hyperparam_counts = defaultdict(int)  # Para almacenar los conteos de hiperparámetros del bucle interno
        inner_fold_counter = 1  

        for train_idx_inner, val_idx_inner in inner_cv.split(X.iloc[train_idx_outer], y.iloc[train_idx_outer]):
            X_train_inner, y_train_inner = X.iloc[train_idx_inner], y.iloc[train_idx_inner]
            X_val_inner, y_val_inner = X.iloc[val_idx_inner], y.iloc[val_idx_inner]
        
            selected_features = X.columns
            
            if len(selected_features) == 0:
                inner_results.append({
                    'Fold': f'outer_{outer_fold_counter}_inner_{inner_fold_counter}',
                    'Features': [], 
                    'MSE': None,
                    'Status': 'Failed due to 0 features selected'
                })
                continue
            
            # Entrenamiento del modelo y ajuste de hiperparámetros en el bucle interno
            best_model, model_params = classifier(X_train_inner[selected_features], y_train_inner, cls, inner_cv)
            
            # Actualizar los conteos de características e hiperparámetros
            for feature in selected_features:
                feature_counts[feature] += 1
            for key, value in model_params.items():
                hyperparam_counts[key + "_" + str(value)] += 1
                
            # Evaluar el modelo en el bucle interno
            y_pred_inner = best_model.predict(X_val_inner[selected_features])
            mse_inner = mean_squared_error(y_val_inner, y_pred_inner)
            
            inner_results.append({
                'Fold': 'outer_' + str(outer_fold_counter) + '_inner_' + str(inner_fold_counter),
                'Features': selected_features, 
                'Model_Params': model_params,
                'MSE': mse_inner
            })
        
            inner_fold_counter += 1  # Incrementar el contador del bucle interno
        
        most_common_features = [feature for feature, _ in Counter(feature_counts).most_common(5)]
        most_common_hyperparams = process_most_common_hyperparams(hyperparam_counts)

        X_train_outer, y_train_outer = X.iloc[train_idx_outer], y.iloc[train_idx_outer]
        X_val_outer, y_val_outer = X.iloc[val_idx_outer], y.iloc[val_idx_outer]

        print("Most common features:", most_common_features)
        print("Training data shape:", X_train_outer[most_common_features].shape)
        print("Validation data shape:", X_val_outer[most_common_features].shape)
        
        final_model, _ = classifier(X_train_outer[most_common_features], y_train_outer, cls, inner_cv, params=most_common_hyperparams)
        
        final_model.fit(X_train_outer[most_common_features], y_train_outer.values.ravel())
        y_pred_outer = final_model.predict(X_val_outer[most_common_features])
        mse_outer = mean_squared_error(y_val_outer, y_pred_outer)
                
        outer_results.append({
            'Fold': 'outer_fold_' + str(outer_fold_counter),
            'Features': most_common_features,
            'Model_Params': most_common_hyperparams,
            'MSE': mse_outer
        })
        
        outer_fold_counter += 1  # Incrementar el contador del bucle interno

    return pd.DataFrame(inner_results), pd.DataFrame(outer_results)


class NumpyEncoder(json.JSONEncoder):  #we need this fro xgboost results. Error "object of type float32 is not JSON serializable"
    def default(self, obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)

def get_results(X, y, outer_results_df, odir, agg, cls, sel):
    # Determinar las características seleccionadas más frecuentemente de los pliegues externos
    all_features = [feat for sublist in outer_results_df['Features'].tolist() for feat in sublist]
    most_common_features = [feature for feature, _ in Counter(all_features).most_common(5)]
    
    # Determinar los hiperparámetros más elegidos de los resultados externos
    outer_results_df['Params_String'] = outer_results_df['Model_Params'].astype(str)

    most_common_params_string = Counter(outer_results_df['Params_String']).most_common(1)[0][0]
    most_common_hyperparams = eval(most_common_params_string)    
    
    # Entrenar un modelo final con todos los datos usando las características e hiperparámetros más comunes
    if cls == "lgbm":
        final_model = LGBMRegressor(n_estimators=most_common_hyperparams['n_estimators'], 
                                    max_depth=most_common_hyperparams['max_depth'], 
                                    learning_rate=most_common_hyperparams['learning_rate'], 
                                    subsample=most_common_hyperparams['subsample'], 
                                    random_state=123)
    elif cls == "randomf":
        final_model = RandomForestRegressor(n_estimators=most_common_hyperparams['n_estimators'], 
                                            max_depth=most_common_hyperparams['max_depth'], 
                                            random_state=123)
    elif cls == "xgb":
        final_model = XGBRegressor(eval_metric="rmse", use_label_encoder=False, 
                                   n_estimators=most_common_hyperparams['n_estimators'], 
                                   max_depth=most_common_hyperparams['max_depth'], 
                                   learning_rate=most_common_hyperparams['learning_rate'], 
                                   random_state=123)
   
    final_model.fit(X[most_common_features], y.values.ravel()) # Entrenamiento con todos mis datos
    
    # Calcular el promedio y la desviación estándar de MSEs de los pliegues externos para conocer el rendimiento estimado
    avg_mse = outer_results_df['MSE'].mean()
    std_mse = outer_results_df['MSE'].std()    
    
    results = {
        'regressor': cls,
        'selector': sel,
        'feature_aggregation': agg,
        'average_mse': avg_mse,
        'std_dev_mse': std_mse,
        'hyperparameters': most_common_hyperparams,
        'features': most_common_features
    }
    plt.figure()
    plt.title('SHAP values for the regression model')
    explainer = shap.Explainer(final_model, X[most_common_features])
    shap_values = explainer.shap_values(X[most_common_features])
    shap.summary_plot(shap_values, X[most_common_features], feature_names=most_common_features)
    
    if cls in ["randomf", "xgb", "lgbm"]:
        results['feature_importances'] = dict(zip(most_common_features, final_model.feature_importances_))
        
    final_model_path = os.path.join(odir, "results_"+agg+"_"+cls+"_"+sel+".json")
    with open(final_model_path, 'w') as f:
        json.dump(results, f, cls=NumpyEncoder, indent=4)


def main(argv):
    data = argv[0]
    cls = argv[2]
    energy = argv[1]
    name = data + '_' + energy
    rdir = 'C:\\Users\\lclai\\Desktop\\LDIG\\DB\\training\\' + name + '.csv'
   
    odir = 'C:\\Users\\lclai\\Desktop\\' + name


    os.makedirs(odir, exist_ok=True)
    df = pd.read_csv(rdir)
    df = df.filter(regex='^(?!Unnamed)')
    cols = [col for col in df.columns if 'pubchem' not in col and 'Eat' not in col and 'mf' not in col]
    X = df[cols]
    y = df[['Eat']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    print("Scaled data shape:", X_scaled.shape)
    print("Target data shape:", y.shape)
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=123)
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=123)
    inner_results_df, outer_results_df = nested_cv(X_scaled, y, outer_cv, inner_cv, cls, data)
    final_results = get_results(X, y, outer_results_df, odir, energy, cls, data)

if __name__ == "__main__":
    main(sys.argv[1:])
    print("Finish!")

# To run the code use the following structure    
# python train_radiomics.py anova logisticregression largest

