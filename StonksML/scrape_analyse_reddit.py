from scrape_reddit import ScrapeReddit
from reddit_sentiment_analysis import reddit_sentiment_analysis


def scrape_analyse_reddit():
    ScrapeReddit.scrape()
    reddit_sentiment_analysis()


if __name__ == "__main__":
    scrape_analyse_reddit()