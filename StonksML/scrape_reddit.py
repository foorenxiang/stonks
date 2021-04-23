# Modified from https://medium.com/analytics-vidhya/praw-a-python-package-to-scrape-reddit-post-data-b759a339ed9a

import logging
from pathlib import Path
from dotenv import load_dotenv
from os import environ as env
import praw
from datetime import datetime
import pandas as pd
from joblib import dump

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler(
            filename=Path(__file__).resolve().parent / "reddit_scrape.log", mode="w"
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def scrape_reddit():
    current_directory = Path(__file__).resolve().parent
    save_directory = current_directory.parent / "datasets" / "reddit_dump"

    try:
        save_directory.mkdir()
    except FileExistsError:
        pass

    load_dotenv()

    REDDIT_PERSONAL_USE_SCRIPT = env["REDDIT_PERSONAL_USE_SCRIPT"]
    REDDIT_SECRET = env["REDDIT_SECRET"]
    REDDIT_USERNAME = env["REDDIT_USERNAME"]
    REDDIT_PASSWORD = env["REDDIT_PASSWORD"]
    REDDIT_USERAGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.85 Safari/537.36"

    reddit = praw.Reddit(
        client_id=REDDIT_PERSONAL_USE_SCRIPT,
        client_secret=REDDIT_SECRET,
        usernme=REDDIT_USERNAME,
        password=REDDIT_PASSWORD,
        user_agent=REDDIT_USERAGENT,
    )

    subred_list = []
    author_list = []
    id_list = []
    link_flair_text_list = []
    num_comments_list = []
    score_list = []
    title_list = []
    upvote_ratio_list = []

    subreddit_list = [
        "wallstreetbets",
        "ocugen",
        "teslainvestorsclub",
        "SPACs",
        "worldnews",
        "singapore",
    ]

    logger.info(f"Started scrapping at {datetime.now()}")

    for subred in subreddit_list:
        logger.info(f"Starting to scrape {subred.upper()}")
        subreddit = reddit.subreddit(subred)
        hot_post = subreddit.hot(limit=10000)

        for sub in hot_post:
            subred_list.append(subred)
            author_list.append(sub.author)
            # id_list.append(sub.id)
            link_flair_text_list.append(sub.link_flair_text)
            num_comments_list.append(sub.num_comments)
            score_list.append(sub.score)
            title_list.append(sub.title)
            upvote_ratio_list.append(sub.upvote_ratio)

        logger.info(f"Scraped {len(title_list)} posts from {subred}")

    df = pd.DataFrame(
        {
            "Subreddit": subred_list,
            "Title": title_list,
            "Count_of_Comments": num_comments_list,
            "Upvote_Count": score_list,
            "Upvote_Ratio": upvote_ratio_list,
            "Flair": link_flair_text_list,
            "Author": author_list,
        }
    )

    df.to_csv(save_directory / "reddit_dataset.csv", index=False)

    dump(df, save_directory / "reddit_dataset.joblib")


if __name__ == "__main__":
    scrape_reddit()
    from reddit_sentiment_analysis import reddit_sentiment_analysis

    reddit_sentiment_analysis()