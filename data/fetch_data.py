import yfinance as yf

def fetch_stock_data(ticker):
    """
    Fetch historical stock data for a given ticker.
    :param ticker: Stock ticker symbol (e.g., "AAPL")
    :return: DataFrame with stock data
    """
    try:
        stock_data = yf.Ticker(ticker)
        df = stock_data.history(period="max")  # Fetch all available data
        if df.empty:
            raise ValueError(f"No data found for ticker: {ticker}")
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.index = df.index.date  # Set index to date for consistency
        return df
    except Exception as e:
        print(f"Error fetching stock data for {ticker}: {e}")
        return None
