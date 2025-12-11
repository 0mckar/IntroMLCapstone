import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers
from sklearn.preprocessing import StandardScaler
import time
import warnings
warnings.filterwarnings('ignore')

class NeuralNetworkRegressor:
    def __init__(self, input_dim=None, layers_config=[64, 32, 16], 
                 dropout_rate=0.2, learning_rate=0.001, l2_reg=0.001):

        self.input_dim = input_dim
        self.layers_config = layers_config
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.model = None
        self.scaler = StandardScaler()
        self.history = None
        
    def _build_model(self, input_dim):
        model = keras.Sequential()
        
        model.add(layers.Input(shape=(input_dim,)))
        
        for i, units in enumerate(self.layers_config):
            model.add(layers.Dense(
                units, 
                activation='relu',
                kernel_regularizer=regularizers.l2(self.l2_reg)
            ))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(self.dropout_rate))
        
        model.add(layers.Dense(1, activation='linear'))
        
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def fit(self, x_train, y_train, x_val=None, y_val=None, 
            epochs=100, batch_size=32, verbose=1):

        print(f"\nTraining Neural Network (Architecture: {self.layers_config})...")
        start_time = time.time()

        if np.any(np.isnan(x_train)) or np.any(np.isnan(y_train)):
            print("WARNING: NaN values detected in training data!")
            x_train = np.nan_to_num(x_train)
            y_train = np.nan_to_num(y_train)
        
        x_train_scaled = self.scaler.fit_transform(x_train)
        
        if self.model is None:
            self.model = self._build_model(x_train_scaled.shape[1])
        
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss' if x_val is not None else 'loss',
                patience=20,
                restore_best_weights=True,
                verbose=0
            )
        ]
        
        if x_val is not None and y_val is not None:
            x_val_scaled = self.scaler.transform(x_val)
            validation_data = (x_val_scaled, y_val)
        else:
            validation_data = None
        
        self.history = self.model.fit(
            x_train_scaled, y_train,
            validation_data=validation_data,
            validation_split=0.2 if validation_data is None else None,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=verbose
        )
        
        training_time = time.time() - start_time
        
        print(f"\nTraining completed in {training_time:.2f} seconds")
        
        return {
            'model': self.model,
            'history': self.history,
            'training_time': training_time
        }
    
    def predict(self, x_test):
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        x_scaled = self.scaler.transform(x_test)
        predictions = self.model.predict(x_scaled, verbose=0)
        return predictions.flatten()
    
    def score(self, x_test, y_test):
        from sklearn.metrics import r2_score
        y_pred = self.predict(x_test)
        return r2_score(y_test, y_pred)