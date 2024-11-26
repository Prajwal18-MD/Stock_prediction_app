import streamlit as st
import pandas as pd
import datetime as dt
from models.lstm import LSTMModel
from models.gru import GRUModel
from models.cnn import CNNModel
from models.transformer import TransformerModel
from models.ensemble import EnsembleModel
from data.fetch_data import fetch_stock_data
from data.sentiment_analysis import get_news_sentiment
from data.indicators import calculate_technical_indicators
from portfolio.optimization import MarkowitzOptimization
from portfolio.risk_management import stop_loss_take_profit, calculate_var
from strategies.put_call_parity import put_call_parity_strategy
from strategies.straddle import straddle_strategy
from strategies.strangle import strangle_strategy
from utils.logging_utils import setup_logging
from utils.interpretability import explain_predictions
from utils.news_ticker import display_news_ticker

# Setup logging
setup_logging()

# App title and description
st.title("Advanced Stock Prediction App 📈")
st.markdown("""
    This application allows users to predict stock prices using various advanced models and strategies.
    Enter the stock ticker and target date, then press the 'Predict' button to see the results.
""")

# User input
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL):")
target_date = st.date_input("Enter Target Date:", min_value=dt.date.today())

if st.button("Predict"):
    if ticker and target_date:
        try:
            # Fetch stock data
            data = fetch_stock_data(ticker)
            if data is None or data.empty:
                st.error("Could not fetch stock data. Please check the ticker and try again.")
                st.stop()

            # Validate target date
            if target_date < data.index.min() or target_date > data.index.max():
                st.error("Target date is out of range. Please select a valid date.")
                st.stop()

            # Calculate technical indicators
            data = calculate_technical_indicators(data)

            # Get sentiment analysis
            sentiment_score = get_news_sentiment(ticker)
            if sentiment_score is not None:
                st.write(f"Sentiment Score: {sentiment_score}")
            else:
                st.warning("No sentiment data available.")

            # Prepare data for models
            data = data.dropna()
            features = data.drop(['Close'], axis=1)
            target = data['Close']

            # Initialize and train models
            lstm_model = LSTMModel()
            gru_model = GRUModel()
            cnn_model = CNNModel()
            transformer_model = TransformerModel()
            ensemble_model = EnsembleModel([lstm_model, gru_model, cnn_model])

            lstm_model.train(features, target)
            gru_model.train(features, target)
            cnn_model.train(features, target)
            transformer_model.train(features, target)
            ensemble_model.train(features, target)

            # Predict prices
            lstm_pred = lstm_model.predict(target_date)
            gru_pred = gru_model.predict(target_date)
            cnn_pred = cnn_model.predict(target_date)
            transformer_pred = transformer_model.predict(target_date)
            ensemble_pred = ensemble_model.predict(target_date)

            # Display predictions
            st.write(f"LSTM Prediction: {lstm_pred}")
            st.write(f"GRU Prediction: {gru_pred}")
            st.write(f"CNN Prediction: {cnn_pred}")
            st.write(f"Transformer Prediction: {transformer_pred}")
            st.write(f"Ensemble Prediction: {ensemble_pred}")

            # Plot predictions
            st.line_chart(data['Close'])

            # Strategies
            st.markdown("### Trading Strategies")
            st.write("**Put-Call Parity Strategy:**")
            st.write(put_call_parity_strategy(ticker, target_date))
            st.write("**Straddle Strategy:**")
            st.write(straddle_strategy(ticker, target_date))
            st.write("**Strangle Strategy:**")
            st.write(strangle_strategy(ticker, target_date))

            # Portfolio Optimization
            st.markdown("### Portfolio Optimization")
            try:
                st.write("**Markowitz Optimization:**")
                st.write(MarkowitzOptimization(data))
            except Exception as e:
                st.error(f"Error in portfolio optimization: {e}")

            # Risk Management
            st.markdown("### Risk Management")
            st.write("**Stop-Loss/Take-Profit Strategy:**")
            st.write(stop_loss_take_profit(data))
            st.write("**Value at Risk (VaR):**")
            st.write(calculate_var(data))

            # Explain predictions
            st.markdown("### Model Interpretability")
            st.write("**SHAP Explanations:**")
            st.write(explain_predictions(ensemble_model, features))

            # Display live news ticker
            st.markdown("### Live Stock News")
            display_news_ticker(ticker)

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.error("Please enter both stock ticker and target date.")
