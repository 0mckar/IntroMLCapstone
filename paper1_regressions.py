import numpy as np
import pandas as pd
import math
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

def mape(actual, predict):
    tmp, n = 0.0, 0
    for i in range(0, len(actual)):
        if actual[i] != 0:
            tmp += math.fabs(actual[i]-predict[i])/actual[i]
            n += 1
    return float((tmp/n)*100)

def run_paper1_models(X_train, y_train, X_test, y_test):
    results = {}
    
    print("\n" + "="*60)
    print("PAPER 1: COMPREHENSIVE REGRESSION ANALYSIS")
    print("="*60)
    
    # Ensure inputs are numpy arrays
    if not isinstance(X_train, np.ndarray):
        X_train = X_train.values
    if not isinstance(y_train, np.ndarray):
        y_train = y_train.values.ravel()
    if not isinstance(X_test, np.ndarray):
        X_test = X_test.values
    if not isinstance(y_test, np.ndarray):
        y_test = y_test.values.ravel()
    
    alphasr = [0.001, 0.01, 0.1, 0.5, 1, 10, 100, 1000]
    
    print("\n1. Ridge Regression (with CV)...")
    RidgeCV_model = RidgeCV(cv=5, alphas=alphasr).fit(X_train, y_train)
    alpha_ridge = RidgeCV_model.alpha_
    ridge_reg = Ridge(alpha=alpha_ridge)
    ridge_reg.fit(X_train, y_train)
    ridge_predictions = ridge_reg.predict(X_test)
    
    results['ridge'] = {
        'model': ridge_reg,
        'predictions': ridge_predictions,
        'alpha': alpha_ridge
    }
    
    print("\n2. Linear Regression...")
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    lin_predictions = lin_reg.predict(X_test)
    
    results['linear'] = {
        'model': lin_reg,
        'predictions': lin_predictions
    }
    
    print("\n3. Decision Tree...")
    dt_reg = DecisionTreeRegressor(random_state=10,
                                   max_depth=13,
                                   min_samples_leaf=4,
                                   min_samples_split=4,
                                   max_features='sqrt')  # Changed from 'auto'
    dt_reg.fit(X_train, y_train)
    dt_predictions = dt_reg.predict(X_test)
    
    results['decision_tree'] = {
        'model': dt_reg,
        'predictions': dt_predictions
    }
    
    print("\n4. Random Forest...")
    rf_reg = RandomForestRegressor(random_state=42)
    rf_reg.fit(X_train, y_train)
    rf_predictions = rf_reg.predict(X_test)
    
    results['random_forest'] = {
        'model': rf_reg,
        'predictions': rf_predictions
    }
    
    print("\n5. AdaBoost...")
    ada_reg = AdaBoostRegressor(random_state=0, n_estimators=100)
    ada_reg.fit(X_train, y_train)
    ada_predictions = ada_reg.predict(X_test)
    
    results['adaboost'] = {
        'model': ada_reg,
        'predictions': ada_predictions
    }
    
    print("\n6. XGBoost...")
    XG_reg = xgb.XGBRegressor(n_estimators=1000, 
                              max_depth=7, 
                              eta=0.1, 
                              subsample=0.2, 
                              colsample_bytree=0.8,
                              random_state=42)
    XG_reg.fit(X_train, y_train)
    xgb_predictions = XG_reg.predict(X_test)
    
    results['xgboost'] = {
        'model': XG_reg,
        'predictions': xgb_predictions
    }
    
    for model_name, model_data in results.items():
        predictions = model_data['predictions']
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        mape_val = mape(y_test, predictions)  # y_test is now numpy array
        r2 = r2_score(y_test, predictions)
        
        n = len(y_test)
        p = X_train.shape[1]
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        
        results[model_name]['metrics'] = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape_val,
            'r2': r2,
            'adjusted_r2': adjusted_r2
        }
        
        model_display_name = model_name.replace('_', ' ').title()
        print(f"\n{model_display_name}:")
        print(f"  RMSE: {rmse:.2f}, R²: {r2:.4f}, MAPE: {mape_val:.2f}%")
    
    return results

def get_paper1_metrics_summary(results):
    summary = []
    for model_name, model_data in results.items():
        metrics = model_data['metrics']
        model_display_name = model_name.replace('_', ' ').title()
        summary.append({
            'Model': f"{model_display_name} (Paper 1)",
            'mse': metrics['mse'],
            'rmse': metrics['rmse'],
            'mae': metrics['mae'],
            'mape': metrics['mape'],
            'r2': metrics['r2'],
            'adjusted_r2': metrics['adjusted_r2']
        })
    return summary

def run_paper1_demo():
    print("Paper 1 Regression Module Loaded Successfully")
    print("Available functions: run_paper1_models, get_paper1_metrics_summary")
    
if __name__ == "__main__":
    run_paper1_demo()