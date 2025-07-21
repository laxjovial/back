from textblob import TextBlob

class SentimentAnalysisTool:
    def analyze_sentiment(self, text: str) -> dict:
        """
        Analyzes the sentiment of a given text.

        :param text: The text to analyze.
        :return: A dictionary with the polarity and subjectivity.
        """
        blob = TextBlob(text)
        return {"polarity": blob.sentiment.polarity, "subjectivity": blob.sentiment.subjectivity}
