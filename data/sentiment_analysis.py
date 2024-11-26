from textblob import TextBlob
from newspaper import Article
import datetime

class SentimentAnalysis:
    def __init__(self, news_url):
        self.news_url = news_url

    def analyze_sentiment(self):
        """
        Perform sentiment analysis on a news article.
        :return: Sentiment polarity (score between -1 and 1).
        """
        try:
            article = Article(self.news_url)
            article.download()
            article.parse()
            article.nlp()
            analysis = TextBlob(article.text)
            return analysis.sentiment.polarity
        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
            return None
