import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def run_paper2_xgboost(X, y, test_size=0.2, random_state=42):
    print("\n" + "="*60)
    print("PAPER 2: XGBOOST OPTIMIZATION STUDY")
    print("="*60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    
    print("\n1. Base XGBoost Model...")
    xgb_mod = xgb.XGBRegressor(random_state=random_state)
    xgb_mod.fit(X_train, y_train)
    xgb_pred = xgb_mod.predict(X_test)
    
    base_metrics = calculate_metrics(y_test, xgb_pred, "Base XGBoost")
    
    print("\n2. Optimized XGBoost Model...")
    xgb_opt = xgb.XGBRegressor(objective="reg:squarederror", 
                               max_depth=10,
                               gamma=0.001,
                               min_child_weight=50,
                               subsample=1,
                               random_state=random_state)
    xgb_opt.fit(X_train, y_train)
    xgb_opt_pred = xgb_opt.predict(X_test)
    
    opt_metrics = calculate_metrics(y_test, xgb_opt_pred, "Optimized XGBoost")
    
    print("\n3. Feature Importance Analysis...")
    feature_important = xgb_opt.get_booster().get_score(importance_type='total_gain')
    keys = list(feature_important.keys())
    values = list(feature_important.values())
    
    feature_importance_df = pd.DataFrame(data=values, index=keys, columns=["score"])
    feature_importance_df = feature_importance_df.sort_values(by="score", ascending=True)
    
    top_features = feature_importance_df.nlargest(10, columns="score")
    
    print("\nTop 10 Features by Importance:")
    for idx, (feature, score) in enumerate(zip(top_features.index, top_features['score']), 1):
        print(f"  {idx:2d}. {feature}: {score:.4f}")
    
    print("\n4. Cross-Validation Results...")
    scores_cvs = cross_val_score(xgb_opt, X, y, scoring='r2', cv=5)
    cv_accuracy = scores_cvs.mean() * 100
    cv_std = scores_cvs.std() * 100
    
    print(f"Cross-validation R²: {cv_accuracy:.2f}% (±{cv_std:.2f}%)")
    
    results = {
        'base_model': xgb_mod,
        'optimized_model': xgb_opt,
        'base_predictions': xgb_pred,
        'optimized_predictions': xgb_opt_pred,
        'base_metrics': base_metrics,
        'optimized_metrics': opt_metrics,
        'feature_importance': feature_importance_df,
        'cv_accuracy': cv_accuracy,
        'cv_std': cv_std,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
    
    return results

def calculate_metrics(y_true, y_pred, model_name=""):
    y_true = y_true.flatten() if hasattr(y_true, 'flatten') else y_true
    y_pred = y_pred.flatten() if hasattr(y_pred, 'flatten') else y_pred
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    n = len(y_true)
    p = 1
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    mape_val = calculate_mape(y_true, y_pred)
    
    if model_name:
        print(f"\n{model_name} Metrics:")
        print(f"  MSE: {mse:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAE: {mae:.2f}")
        print(f"  MAPE: {mape_val:.2f}%")
        print(f"  R²: {r2:.4f}")
        print(f"  Adjusted R²: {adjusted_r2:.4f}")
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape_val,
        'r2': r2,
        'adjusted_r2': adjusted_r2
    }

def calculate_mape(y_true, y_pred):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    mask = y_true != 0
    if np.sum(mask) == 0:
        return np.nan
    
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def plot_paper2_results(results):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    y_test = results['y_test']
    opt_pred = results['optimized_predictions']
    
    axes[0, 0].scatter(y_test, opt_pred, alpha=0.5, color='crimson')
    p1 = max(max(opt_pred), max(y_test))
    p2 = min(min(opt_pred), min(y_test))
    axes[0, 0].plot([p1, p2], [p1, p2], 'b-')
    axes[0, 0].set_xlabel('Actual Values')
    axes[0, 0].set_ylabel('Predicted Values')
    axes[0, 0].set_title('Actual vs Predicted (Optimized XGBoost)')
    axes[0, 0].grid(True, alpha=0.3)
    
    residuals = y_test - opt_pred
    axes[0, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Residuals')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Residual Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    top_features = results['feature_importance'].nlargest(10, columns="score")
    axes[1, 0].barh(range(len(top_features)), top_features['score'])
    axes[1, 0].set_yticks(range(len(top_features)))
    axes[1, 0].set_yticklabels(top_features.index)
    axes[1, 0].set_xlabel('Importance Score')
    axes[1, 0].set_title('Top 10 Feature Importance')
    
    base_pred = results['base_predictions']
    opt_pred = results['optimized_predictions']
    axes[1, 1].scatter(y_test, base_pred, alpha=0.5, label='Base XGBoost', color='blue')
    axes[1, 1].scatter(y_test, opt_pred, alpha=0.5, label='Optimized XGBoost', color='red')
    axes[1, 1].plot([p1, p2], [p1, p2], 'k--', alpha=0.5)
    axes[1, 1].set_xlabel('Actual Values')
    axes[1, 1].set_ylabel('Predicted Values')
    axes[1, 1].set_title('Base vs Optimized XGBoost')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def get_paper2_metrics_summary(results):
    summary = []
    
    summary.append({
        'Model': 'XGBoost Base (Paper 2)',
        'mse': results['base_metrics']['mse'],
        'rmse': results['base_metrics']['rmse'],
        'mae': results['base_metrics']['mae'],
        'mape': results['base_metrics']['mape'],
        'r2': results['base_metrics']['r2'],
        'adjusted_r2': results['base_metrics']['adjusted_r2']
    })
    
    summary.append({
        'Model': 'XGBoost Optimized (Paper 2)',
        'mse': results['optimized_metrics']['mse'],
        'rmse': results['optimized_metrics']['rmse'],
        'mae': results['optimized_metrics']['mae'],
        'mape': results['optimized_metrics']['mape'],
        'r2': results['optimized_metrics']['r2'],
        'adjusted_r2': results['optimized_metrics']['adjusted_r2']
    })
    
    return summary

def run_paper2_demo():
    print("Paper 2 XGBoost Module Loaded Successfully")
    print("Available functions: run_paper2_xgboost, get_paper2_metrics_summary, plot_paper2_results")
    
if __name__ == "__main__":
    run_paper2_demo()