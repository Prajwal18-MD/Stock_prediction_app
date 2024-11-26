from tensorflow.keras.layers import Conv1D, Flatten, Dense
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class CNNModel:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()

    def train(self, features, target):
        data_scaled = self.scaler.fit_transform(features)
        X_train, y_train = data_scaled[:-1], target[1:]
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        self.model = Sequential()
        self.model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
        self.model.add(Flatten())
        self.model.add(Dense(units=50, activation='relu'))
        self.model.add(Dense(units=1))

        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

    def predict(self, future_data):
        scaled_data = self.scaler.transform(future_data)
        scaled_data = np.reshape(scaled_data, (scaled_data.shape[0], scaled_data.shape[1], 1))
        predictions = self.model.predict(scaled_data)
        return self.scaler.inverse_transform(predictions)
