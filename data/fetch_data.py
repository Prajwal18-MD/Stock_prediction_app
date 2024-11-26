import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker):
    """
    Fetch historical stock data from the stock's inception until today.
    :param ticker: Stock ticker symbol.
    :return: DataFrame containing stock data.
    """
    try:
        # Use period="max" to fetch data from the earliest available date
        stock_data = yf.download(ticker, period="max")
        
        # Select relevant columns
        stock_data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Reset the index to ensure it's properly formatted
        stock_data.reset_index(inplace=True)
        
        return stock_data
    except Exception as e:
        raise ValueError(f"Error fetching data for {ticker}: {e}")
