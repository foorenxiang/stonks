from scrape_reddit import scrape
from reddit_sentiment_analysis import reddit_sentiment_analysis


def scrape_analyse_reddit():
    scrape()
    reddit_sentiment_analysis()


if __name__ == "__main__":
    scrape_analyse_reddit()