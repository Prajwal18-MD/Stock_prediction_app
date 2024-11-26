import pandas as pd

class TechnicalIndicators:
    def __init__(self, data):
        self.data = data

    def calculate_bollinger_bands(self, window=20):
        self.data['SMA'] = self.data['Close'].rolling(window=window).mean()
        self.data['Upper Band'] = self.data['SMA'] + (self.data['Close'].rolling(window=window).std() * 2)
        self.data['Lower Band'] = self.data['SMA'] - (self.data['Close'].rolling(window=window).std() * 2)
        return self.data[['SMA', 'Upper Band', 'Lower Band']]

    def calculate_rsi(self, window=14):
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))
        return self.data['RSI']

    def calculate_macd(self, short_window=12, long_window=26, signal_window=9):
        self.data['12_EMA'] = self.data['Close'].ewm(span=short_window, adjust=False).mean()
        self.data['26_EMA'] = self.data['Close'].ewm(span=long_window, adjust=False).mean()
        self.data['MACD'] = self.data['12_EMA'] - self.data['26_EMA']
        self.data['Signal Line'] = self.data['MACD'].ewm(span=signal_window, adjust=False).mean()
        return self.data[['MACD', 'Signal Line']]

    def calculate_ema(self, span=20):
        self.data['EMA'] = self.data['Close'].ewm(span=span, adjust=False).mean()
        return self.data['EMA']

    def calculate_stochastic_oscillator(self, k_window=14, d_window=3):
        self.data['Low_K'] = self.data['Low'].rolling(window=k_window).min()
        self.data['High_K'] = self.data['High'].rolling(window=k_window).max()
        self.data['%K'] = 100 * ((self.data['Close'] - self.data['Low_K']) / (self.data['High_K'] - self.data['Low_K']))
        self.data['%D'] = self.data['%K'].rolling(window=d_window).mean()
        return self.data[['%K', '%D']]
