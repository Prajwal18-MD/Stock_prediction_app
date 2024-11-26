import pandas as pd
import numpy as np

def calculate_technical_indicators(data):
    """
    Calculate various technical indicators for the stock data.
    :param data: DataFrame containing stock data with columns 'Close', 'High', 'Low', and 'Volume'.
    :return: DataFrame with additional columns for technical indicators.
    """
    try:
        # Ensure required columns are present
        required_columns = ['Close', 'High', 'Low', 'Volume']
        if not all(col in data.columns for col in required_columns):
            raise ValueError("Missing required columns in stock data")

        # Moving Average
        data['SMA_20'] = data['Close'].rolling(window=20).mean()  # Simple Moving Average (20 days)
        data['SMA_50'] = data['Close'].rolling(window=50).mean()  # Simple Moving Average (50 days)

        # Relative Strength Index (RSI)
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        data['Bollinger_Mid'] = data['Close'].rolling(window=20).mean()
        data['Bollinger_Std'] = data['Close'].rolling(window=20).std()
        data['Bollinger_Upper'] = data['Bollinger_Mid'] + (2 * data['Bollinger_Std'])
        data['Bollinger_Lower'] = data['Bollinger_Mid'] - (2 * data['Bollinger_Std'])

        # MACD (Moving Average Convergence Divergence)
        data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
        data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

        # Drop NA values after calculations to ensure consistency
        data = data.dropna()

        return data
    except Exception as e:
        print(f"Error calculating technical indicators: {e}")
        return data
