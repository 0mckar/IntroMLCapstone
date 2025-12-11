import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score, KFold
import time
import warnings
warnings.filterwarnings('ignore')

class LinearRegressionModel:
    def __init__(self, model_type='linear', alpha=1.0):
        self.model_type = model_type
        self.alpha = alpha
        self.model = None
        self.intercept_ = None
        self.coef_ = None
        
        if model_type == 'linear':
            self.model = LinearRegression()
        elif model_type == 'ridge':
            self.model = Ridge(alpha=alpha, random_state=42)
        elif model_type == 'lasso':
            self.model = Lasso(alpha=alpha, max_iter=10000, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def fit(self, x_train, y_train, cv_folds=5):

        print(f"\nTraining {self.model_type.upper()} Regression...")
        start_time = time.time()
        
        self.model.fit(x_train, y_train)
        training_time = time.time() - start_time
        
        self.intercept_ = self.model.intercept_
        self.coef_ = self.model.coef_
        
        cv_scores = cross_val_score(self.model, x_train, y_train, 
                                   cv=cv_folds, scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores)
        
        print(f"Training completed in {training_time:.2f} seconds")
        print(f"Cross-validation RMSE: {cv_rmse.mean():.2f} (±{cv_rmse.std():.2f})")
        
        return {
            'model': self.model,
            'training_time': training_time,
            'cv_rmse_mean': cv_rmse.mean(),
            'cv_rmse_std': cv_rmse.std()
        }
    
    def predict(self, x_test):
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        return self.model.predict(x_test)
    
    def score(self, x_test, y_test):
        return self.model.score(x_test, y_test)