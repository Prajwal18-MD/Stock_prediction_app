import yfinance as yf
import pandas as pd

class DataFetcher:
    def __init__(self, ticker):
        self.ticker = ticker

    def fetch_data(self, start_date, end_date):
        """
        Fetch historical stock data.
        :param start_date: The start date for fetching data.
        :param end_date: The end date for fetching data.
        :return: DataFrame with stock data.
        """
        stock_data = yf.download(self.ticker, start=start_date, end=end_date)
        stock_data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
        return stock_data

    def fetch_live_price(self):
        """
        Fetch the latest stock price.
        :return: Latest price as float.
        """
        stock = yf.Ticker(self.ticker)
        live_price = stock.history(period="1d")['Close'].iloc[-1]
        return live_price
