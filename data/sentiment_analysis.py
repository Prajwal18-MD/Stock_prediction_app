from textblob import TextBlob
from newspaper import Article
import requests

def get_news_sentiment(stock_ticker):
    """
    Fetch news for a stock ticker and analyze its sentiment.
    :param stock_ticker: Stock ticker symbol (e.g., "AAPL")
    :return: Sentiment polarity (-1 to 1) or None if no sentiment could be calculated
    """
    try:
        # Example NewsAPI integration - Replace with your API key
        api_key = "1a7cb714ba044e33b9b42106bb084ce3"  # Add your NewsAPI key here
        url = f"https://newsapi.org/v2/everything?q={stock_ticker}&apiKey={api_key}"
        response = requests.get(url).json()

        articles = response.get("articles", [])
        if not articles:
            print(f"No news articles found for {stock_ticker}")
            return None

        # Analyze the sentiment of the first article
        article_url = articles[0]["url"]
        article = Article(article_url)
        article.download()
        article.parse()
        analysis = TextBlob(article.text)
        return analysis.sentiment.polarity
    except Exception as e:
        print(f"Error in sentiment analysis for {stock_ticker}: {e}")
        return None
