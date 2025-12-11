import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import math
import locale
warnings.filterwarnings('ignore')

try:
    locale.setlocale(locale.LC_ALL, 'turkish')
except:
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

def mape(actual, predict):
    tmp, n = 0.0, 0
    actual = actual.flatten() if hasattr(actual, 'flatten') else actual
    predict = predict.flatten() if hasattr(predict, 'flatten') else predict
    
    for i in range(0, len(actual)):
        if actual[i] != 0:
            tmp += math.fabs(actual[i]-predict[i])/actual[i]
            n += 1
    return float((tmp/n)*100)

def calculate_metrics(y_true, y_pred, model_name=""):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape_val = mape(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    n = len(y_true)
    p = 1  # Assuming 1 predictor for simplicity
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    if model_name:
        print(f"\n{model_name} Results:")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"MAPE: {mape_val:.4f}%")
        print(f"R² Score: {r2:.4f}")
        print(f"Adjusted R²: {adjusted_r2:.4f}")
    
    return {
        'mse': mse, 
        'rmse': rmse, 
        'mae': mae, 
        'mape': mape_val,
        'r2': r2,
        'adjusted_r2': adjusted_r2
    }

def plot_results(y_true, y_pred, model_name, color='blue'):
    plt.figure(figsize=(10, 6))
    
    plt.scatter(y_true, y_pred, alpha=0.5, color=color, label='Predictions')
    
    max_val = max(y_true.max(), y_pred.max())
    min_val = min(y_true.min(), y_pred.min())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    plt.xlabel('Actual SalePrice')
    plt.ylabel('Predicted SalePrice')
    plt.title(f'{model_name}: Actual vs Predicted')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    metrics = calculate_metrics(y_true, y_pred)
    textstr = f'MSE: {metrics["mse"]:.4f}\nRMSE: {metrics["rmse"]:.4f}\nR²: {metrics["r2"]:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    return plt.gcf()

def plot_model_comparison(results_df):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    metrics_to_plot = ['mse', 'rmse', 'mae', 'mape', 'r2', 'adjusted_r2']
    titles = ['MSE (Lower is better)', 'RMSE (Lower is better)', 'MAE (Lower is better)', 
              'MAPE % (Lower is better)', 'R² (Higher is better)', 'Adjusted R² (Higher is better)']
    
    for idx, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
        ax = axes[idx//3, idx%3]
        bars = ax.barh(results_df['Model'], results_df[metric])
        ax.set_xlabel(title)
        ax.set_title(title)
        ax.invert_yaxis()
        
        for bar in bars:
            width = bar.get_width()
            if metric in ['mape']:
                ax.text(width, bar.get_y() + bar.get_height()/2, 
                       f'{width:.2f}%', 
                       ha='left', va='center', fontsize=8)
            else:
                ax.text(width, bar.get_y() + bar.get_height()/2, 
                       f'{width:.4f}', 
                       ha='left', va='center', fontsize=8)
    
    plt.tight_layout()
    return fig

def load_data(filepath='housepnrdenemeee.xlsx'):
    veriler = pd.read_excel(filepath, engine='openpyxl')
    veriler3 = veriler.copy()
    
    columns = [
        "BsmtFullBath", "GarageQual", "GarageCond", "CentralAir", "GarageType",
        "LotArea", "ExterQual", "LotShape", "GarageYrBlt", "HalfBath", "OpenPorchSF",
        "2ndFlrSF", "WoodDeckSF", "BsmtFinType1", "BsmtFinSF1", "FireplaceQu",
        "HeatingQC", "Foundation", "Fireplaces", "MasVnrArea", "YearRemodAdd",
        "YearBuilt", "TotRmsAbvGrd", "FullBath", "1stFlrSF", "TotalBsmtSF",
        "GarageArea", "GarageCars", "GrLivArea", "OverallQual", "SalePrice"
    ]
    
    missing_cols = [col for col in columns if col not in veriler3.columns]
    if missing_cols:
        print(f"Warning: Missing columns: {missing_cols}")
        columns = [col for col in columns if col in veriler3.columns]
    
    veriler3 = veriler3[columns]
    
    return veriler3