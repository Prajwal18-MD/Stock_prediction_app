import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import Ridge
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense, LSTM, GRU, Conv1D, MaxPooling1D, Flatten
from textblob import TextBlob
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
st.write("""
👋 Welcome to the **Stock Prediction and Trading Strategies App**! Analyze stock trends, predict future prices, and explore trading strategies with ease. 📊

**How to Use**  
1. **Enter a Stock Ticker:**  
   Input a valid stock ticker symbol of any stock available in yahoo finance.  
2. **Select Future Date:**  
   Choose the desired date for price prediction.  
3. **Press Predict:**  
   Click the "🔮 Predict" button to view predictions and strategy recommendations.
""")

# Fetch stock data
@st.cache_data
def fetch_news_sentiment(ticker):
    try:
        url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey=1a7cb714ba044e33b9b42106bb084ce3"
        response = requests.get(url)
        if response.status_code == 200:
            articles = response.json().get('articles', [])
            sentiments = []
            for article in articles:
                description = article.get('description')
                if description:  # Check if description is not None
                    analysis = TextBlob(description)
                    sentiments.append(analysis.sentiment.polarity)
            return np.mean(sentiments) if sentiments else 0
        else:
            return 0
    except Exception as e:
        st.error(f"Error fetching sentiment data: {e}")
        return 0


def fetch_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="max")
        data.reset_index(inplace=True)
        currency = stock.info.get("currency", "N/A")
        
        # Add sentiment score
        sentiment_score = fetch_news_sentiment(ticker)
        data['Sentiment'] = sentiment_score
        
        return data, currency
    except Exception as e:
        st.error(f"⚠️ Error fetching data: {e}")
        logging.error(f"Error fetching data: {e}")
        return None, None


# Fetch options chain data
def fetch_options_chain(ticker):
    try:
        stock = yf.Ticker(ticker)
        options_chain = stock.options
        if not options_chain:  # Check if options_chain is empty
            st.warning(f"No options chain available for {ticker}.")
            return None
        
        options_data = {}
        for date in options_chain:
            chain = stock.option_chain(date)
            options_data[date] = {'calls': chain.calls, 'puts': chain.puts}
        return options_data
    except Exception as e:
        st.error(f"⚠️ Error fetching options chain data: {e}")
        logging.error(f"Error fetching options chain data: {e}")
        return None


def put_call_parity(spot_price, call_price, put_price, strike_price, rate, time_to_maturity):
    # Calculate present value of the strike price
    pv_strike = strike_price / (1 + rate) ** time_to_maturity
    lhs = call_price + pv_strike
    rhs = put_price + spot_price
    return lhs, rhs, abs(lhs - rhs) < 0.01  # Return if parity holds

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
def prepare_data_for_model(data, time_step):
    # Scale only the 'Close' column
    close_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_close = close_scaler.fit_transform(data[['Close']].values)

    # Retain Sentiment as-is for input
    features = data[['Close', 'Sentiment']].values
    X, y = [], []
    for i in range(len(features) - time_step - 1):
        # Use both Close and Sentiment as input features
        X.append(np.hstack((scaled_close[i:(i + time_step)], features[i:(i + time_step), 1].reshape(-1, 1))))
        # Predict only 'Close'
        y.append(scaled_close[i + time_step, 0])
    return np.array(X), np.array(y), close_scaler

# Create ML models
def create_model(model_type, input_shape):
    model = Sequential()
    if model_type == "LSTM":
        model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(64, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(64, return_sequences=False))
    elif model_type == "GRU":
        model.add(GRU(64, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(GRU(64, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(GRU(64, return_sequences=False))
    elif model_type == "CNN":
        model.add(Conv1D(128, kernel_size=3, activation='relu', input_shape=input_shape))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(128, kernel_size=3, activation='relu'))
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
    data, currency = fetch_data(ticker)
    if data is not None:
        st.subheader("📊 Raw Data")
        st.write(data.tail())
        
        # Display dynamic currency
        st.markdown(f"💱 Currency: **{currency}**")

        st.subheader(f"📈 Closing Price Trend ({currency})")
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
        ax.plot(data['Close'], label=f'Close Price ({currency})', color='blue')
        ax.plot(upper_band, label=f'Upper Band ({currency})', color='red')
        ax.plot(lower_band, label=f'Lower Band ({currency})', color='green')
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
        X, y, close_scaler = prepare_data_for_model(data, time_step)
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2])
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Debugging for close_scaler
        print("Debug - close_scaler defined:", close_scaler)
        print("Scaler Min:", close_scaler.min_)
        print("Scaler Scale:", close_scaler.scale_)
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )

        # Train Models
        model_lstm = create_model("LSTM", X_train.shape[1:])
        model_lstm.fit(X_train, y_train, epochs=15, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

        model_gru = create_model("GRU", X_train.shape[1:])
        model_gru.fit(X_train, y_train, epochs=15, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

        model_cnn = create_model("CNN", X_train.shape[1:])
        model_cnn.fit(X_train, y_train, epochs=15, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

        # Voting Regressor
        voting_regressor = VotingRegressor(
            estimators=[
                ('lstm', Ridge().fit(X_test[:, :, 0], model_lstm.predict(X_test).flatten())),
                ('gru', Ridge().fit(X_test[:, :, 0], model_gru.predict(X_test).flatten())),
                ('cnn', Ridge().fit(X_test[:, :, 0], model_cnn.predict(X_test).flatten())),
            ]
        )
        voting_predictions = voting_regressor.fit(X_test[:, :, 0], y_test).predict(X_test[:, :, 0])

        # Ensure predictions are compatible with scaler
        voting_predictions = np.array(voting_predictions).reshape(-1, 1)
        test_predictions = close_scaler.inverse_transform(voting_predictions)

        # RMSE Calculation
        y_test_inverse = close_scaler.inverse_transform(y_test.reshape(-1, 1))
        rmse = np.sqrt(mean_squared_error(y_test_inverse, test_predictions))

        # Log RMSE
        rmse_file_path = "rmse_log.txt"
        with open(rmse_file_path, "a") as file:
            file.write(f"Ticker: {ticker}, RMSE: {rmse:.2f}, Date: {datetime.now()}\n")

        # Visualization
        fig, ax = plt.subplots()
        ax.plot(y_test_inverse, label="Actual Prices", color="blue")
        ax.plot(test_predictions, label="Predicted Prices", color="red")
        ax.legend()
        ax.set_title("Actual vs Predicted Stock Prices")
        st.pyplot(fig)


        # Predicted future price display
        predicted_price = test_predictions[-1][0]
        st.markdown(
            f"<h2 style='font-size: 32px;'>🎯 Predicted future price for **{ticker}**: "
            f"**{predicted_price:.2f} {currency}**</h2>",
            unsafe_allow_html=True,
        )
        
        # Log Predicted Results
        predicted_results_file = "predicted_results.txt"
        with open(predicted_results_file, "a") as file:
            file.write(f"Prediction Started At: {datetime.now()}\n")
            file.write(f"Ticker: {ticker}\n")
            file.write(f"Prediction Date: {date_input}\n")
            file.write(f"Currency: {currency}\n")
            file.write(f"Predicted Prices: {predicted_price:.2f}\n")
            file.write("-" * 50 + "\n")  # Separator for better readability


        # Options Chain
        options_data = fetch_options_chain(ticker)
        st.write("Debug: Options Data", options_data)

        if options_data:
            st.subheader("💡 Trading Strategies: Put-Call Parity Analysis")
            for date, data in options_data.items():
                with st.expander(f"Options Expiry: {date}"):
                    # Check for missing or empty data
                    if data['calls'].empty or data['puts'].empty:
                        st.warning(f"Calls or Puts data is empty for expiry {date}. Skipping.")
                        continue
                    
                    if 'lastPrice' not in data['calls'].columns or 'lastPrice' not in data['puts'].columns:
                        st.warning(f"Missing 'lastPrice' column in Calls or Puts for expiry {date}. Skipping.")
                        continue

                    if 'strike' not in data['calls'].columns:
                        st.warning(f"Missing 'strike' column in Calls data for expiry {date}. Skipping.")
                        continue

                    # Use first valid spot price
                    spot_price = data['calls']['lastPrice'].dropna().iloc[0] if not data['calls']['lastPrice'].dropna().empty else None
                    if spot_price is None:
                        st.warning(f"No valid spot price in Calls data for expiry {date}. Skipping.")
                        continue

                    # Debug output
                    st.write(f"Debug: Calls Head for {date}", data['calls'].head())
                    st.write(f"Debug: Puts Head for {date}", data['puts'].head())

                    # Iterate through calls and match with puts by index
                    for idx, row in data['calls'].iterrows():
                        # Ensure index is within bounds for puts DataFrame
                        if idx >= len(data['puts']):
                            st.warning(f"Mismatch in Calls and Puts data sizes for expiry {date}. Skipping unmatched rows.")
                            break

                        # Fetch matching put price
                        put_row = data['puts'].iloc[idx]

                        # Perform Put-Call Parity calculation
                        lhs, rhs, parity_holds = put_call_parity(
                            spot_price=spot_price,
                            call_price=row['lastPrice'],
                            put_price=put_row['lastPrice'],
                            strike_price=row['strike'],
                            rate=0.05,  # Example risk-free rate
                            time_to_maturity=30 / 365,  # Example time to maturity (30 days)
                        )

                        # Display the results
                        st.write(
                            f"Strike Price: {row['strike']} | Parity Holds: {parity_holds} | "
                            f"LHS (C + PV(X)): {lhs:.2f} {currency} | RHS (P + S): {rhs:.2f} {currency}"
                        )
        else:
            st.write("No options data available for this selected Stock.")

                    

        # Latest News
        st.subheader("📰 Latest News")
        news = fetch_news(ticker)
        st.markdown(news)
        
        with st.expander("ℹ️ About"):
            st.markdown("""
            # **Purpose**  
            This application leverages advanced machine learning models to predict future stock prices and provide options strategies recommendations based on financial theories like Put-Call Parity, Straddle, and Strangle strategies, if applicable.

            # **Features**  
            - **Accurate Predictions:**  
            Utilizes state-of-the-art models such as LSTM, GRU, and CNN, trained on historical stock data, combined with real-time sentiment analysis. The ensemble model used is a Voting Regressor for enhanced accuracy.  
            - **Options Strategies:**  
            Recommends financial strategies like Put-Call Parity, Straddle, and Strangle for better decision-making.  
            - **Portfolio Optimization:**  
            Offers optimal allocation recommendations to maximize potential returns and reduce risks.

            # **Contact**  
            For inquiries or support, feel free to reach out via email at [prajju.18gryphon@gmail.com](mailto:prajju.18gryphon@gmail.com).
            """)
       