import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import time
import warnings
warnings.filterwarnings('ignore')

class PolynomialRegressionModel:
    def __init__(self, degree=2, model_type='ridge', alpha=1.0):

        self.degree = degree
        self.model_type = model_type
        self.alpha = alpha
        self.pipeline = None
        self.model = None
        
        self.poly_features = PolynomialFeatures(
            degree=degree, 
            include_bias=False
        )
        
        if model_type == 'ridge':
            base_model = Ridge(alpha=alpha, random_state=42)
        elif model_type == 'lasso':
            base_model = Lasso(alpha=alpha, max_iter=10000, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.pipeline = Pipeline([
            ('poly', self.poly_features),
            ('model', base_model)
        ])
        
        self.model = base_model
    
    def fit(self, x_train, y_train, cv_folds=5):

        print(f"\nTraining Polynomial Regression (Degree={self.degree}, {self.model_type.upper()})...")
        start_time = time.time()
        
        self.pipeline.fit(x_train, y_train)
        training_time = time.time() - start_time
        
        cv_scores = cross_val_score(self.pipeline, x_train, y_train, 
                                   cv=cv_folds, scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores)
        
        print(f"Training completed in {training_time:.2f} seconds")
        print(f"Cross-validation RMSE: {cv_rmse.mean():.2f} (±{cv_rmse.std():.2f})")
        
        return {
            'pipeline': self.pipeline,
            'training_time': training_time,
            'cv_rmse_mean': cv_rmse.mean(),
            'cv_rmse_std': cv_rmse.std()
        }
    
    def predict(self, x_test):
        if self.pipeline is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        return self.pipeline.predict(x_test)
    
    def score(self, x_test, y_test):
        return self.pipeline.score(x_test, y_test)