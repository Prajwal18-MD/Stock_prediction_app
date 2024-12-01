import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Conv1D, MaxPooling1D, Flatten
from statsmodels.tsa.arima.model import ARIMA
import requests

# Set up logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# App Title
st.set_page_config(page_title="📈 Stock Prediction and Trading Strategies", layout="wide")
st.title('📈 Stock Prediction and Trading Strategies App')

# Sidebar for Inputs
st.sidebar.title("📥 User Input")
ticker = st.sidebar.text_input("🔍 Enter the Stock Ticker (e.g., AAPL, TSLA):", value="")
date_input = st.sidebar.date_input("📅 Select the Prediction Date:", datetime.today())
predict_button = st.sidebar.button("🚀 Predict")

# Welcome Message
st.write("👋 Welcome to the **Stock Prediction and Trading Strategies App**! Analyze stock trends, predict future prices, and explore trading strategies with ease. 📊")

# Fetch stock data
@st.cache_data
def fetch_data(ticker):
    try:
        data = yf.download(ticker, start="2000-01-01")
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        st.error(f"⚠️ Error fetching data: {e}")
        logging.error(f"Error fetching data: {e}")
        return None

# Fetch options chain data
def fetch_options_chain(ticker):
    try:
        stock = yf.Ticker(ticker)
        options_chain = stock.options
        options_data = {}
        for date in options_chain:
            calls = stock.option_chain(date).calls
            puts = stock.option_chain(date).puts
            options_data[date] = {'calls': calls, 'puts': puts}
        return options_data
    except Exception as e:
        st.error(f"⚠️ Error fetching options chain data: {e}")
        logging.error(f"Error fetching options chain data: {e}")
        return None

# Technical Indicators
def compute_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_bollinger_bands(data, window=20, num_std_dev=2):
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std_dev)
    lower_band = rolling_mean - (rolling_std * num_std_dev)
    return upper_band, lower_band

def compute_macd(data, fast=12, slow=26, signal=9):
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_signal

# Prepare data for ML models
def prepare_data_for_model(data_close, time_step):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_close)
    
    X, y = [], []
    for i in range(len(scaled_data) - time_step - 1):
        X.append(scaled_data[i:(i + time_step), 0])
        y.append(scaled_data[i + time_step, 0])
    return np.array(X), np.array(y), scaler

# Create ML models
def create_model(model_type, input_shape):
    model = Sequential()
    if model_type == "LSTM":
        model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
        model.add(LSTM(50, return_sequences=False))
    elif model_type == "GRU":
        model.add(GRU(50, return_sequences=True, input_shape=input_shape))
        model.add(GRU(50, return_sequences=False))
    elif model_type == "CNN":
        model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Fetch News
def fetch_news(ticker):
    try:
        url = f"https://finance.yahoo.com/quote/{ticker}?p={ticker}&.tsrc=fin-srch"
        response = requests.get(url)
        if response.status_code == 200:
            return f"[🔗 Latest News for {ticker}]({url})"
        else:
            return "❌ Failed to fetch news."
    except Exception as e:
        return f"⚠️ Error fetching news: {e}"

# Main logic
if ticker and predict_button:
    st.success(f"🔍 Analyzing **{ticker}** | Prediction Date: **{date_input}**")

    # Fetch stock data
    data = fetch_data(ticker)
    if data is not None:
        st.subheader("📊 Raw Data")
        st.write(data.tail())

        st.subheader("📈 Closing Price Trend")
        st.line_chart(data['Close'])

        # Technical Indicators
        st.subheader("📊 Technical Indicators")
        rsi = compute_rsi(data['Close'])
        upper_band, lower_band = compute_bollinger_bands(data['Close'])
        macd, macd_signal = compute_macd(data['Close'])

        st.write("🔵 **RSI (14-Day)**")
        st.line_chart(rsi)
        st.write("🔵 **Bollinger Bands**")
        fig, ax = plt.subplots()
        ax.plot(data['Close'], label='Close Price', color='blue')
        ax.plot(upper_band, label='Upper Band', color='red')
        ax.plot(lower_band, label='Lower Band', color='green')
        ax.legend()
        st.pyplot(fig)

        st.write("🔵 **MACD**")
        fig, ax = plt.subplots()
        ax.plot(macd, label='MACD', color='purple')
        ax.plot(macd_signal, label='Signal Line', color='orange')
        ax.legend()
        st.pyplot(fig)

        # Prediction with ML Models
        st.subheader("🤖 Prediction with ML Models")
        data_close = data['Close'].values.reshape(-1, 1)
        time_step = 60
        X, y, scaler = prepare_data_for_model(data_close, time_step)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        models = ["LSTM", "GRU", "CNN"]
        predictions = []

        for model_type in models:
            model = create_model(model_type, X_train.shape[1:])
            model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
            pred = model.predict(X_test)
            predictions.append(pred)

        # Ensemble Prediction
        ensemble_pred = np.mean(predictions, axis=0)
        test_predictions = scaler.inverse_transform(ensemble_pred)

        # Calculate RMSE and append to file
        rmse = np.sqrt(mean_squared_error(scaler.inverse_transform(y_test.reshape(-1, 1)), test_predictions))
        with open("rmse.txt", "a") as file:
           file.write(f"Ticker: {ticker}, RMSE: {rmse:.2f}\n")


        # Plot predicted vs actual
        st.subheader("🔮 Predicted vs Actual Prices")
        fig, ax = plt.subplots()
        ax.plot(scaler.inverse_transform(y_test.reshape(-1, 1)), label="Actual Price", color="blue")
        ax.plot(test_predictions, label="Predicted Price", color="red")
        ax.set_title("Predicted vs Actual Prices")
        ax.legend()
        st.pyplot(fig)

        # Predicted future price (Improved Font Size)
        predicted_price = test_predictions[-1][0]
        st.markdown(f"<h2 style='font-size: 32px;'>🎯 Predicted future price for **{ticker}**: **{predicted_price:.2f} USD**</h2>", unsafe_allow_html=True)

        # Options Chain
        st.subheader("💡 Trading Strategies: Options Chain")
        options_data = fetch_options_chain(ticker)
        if options_data:
            for date, data in options_data.items():
                with st.expander(f"Options Expiry: {date}"):
                    st.write("📈 Calls")
                    st.write(data['calls'].head())
                    st.write("📉 Puts")
                    st.write(data['puts'].head())

        # Latest News
        st.subheader("📰 Latest News")
        news = fetch_news(ticker)
        st.markdown(news)
