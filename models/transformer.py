import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, MultiHeadAttention, Dropout
from tensorflow.keras import Sequential

class TransformerModel:
    def __init__(self, units=128, num_heads=2, dropout_rate=0.1):
        self.model = None
        self.units = units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.scaler = MinMaxScaler()

    def build_model(self, input_shape):
        inputs = tf.keras.Input(shape=input_shape)
        x = LayerNormalization(epsilon=1e-6)(inputs)
        x = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.units)(x, x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(self.units, activation="relu")(x)
        outputs = Dense(1)(x)
        
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer="adam", loss="mean_squared_error")

    def train(self, features, target):
        data_scaled = self.scaler.fit_transform(features)
        X_train, y_train = data_scaled[:-1], target[1:]
        input_shape = (X_train.shape[1], X_train.shape[2] if len(X_train.shape) > 2 else 1)
        self.build_model(input_shape)
        self.model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

    def predict(self, future_data):
        scaled_data = self.scaler.transform(future_data)
        predictions = self.model.predict(scaled_data)
        return self.scaler.inverse_transform(predictions)
