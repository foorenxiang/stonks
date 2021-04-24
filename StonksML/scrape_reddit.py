# Modified from https://medium.com/analytics-vidhya/praw-a-python-package-to-scrape-reddit-post-data-b759a339ed9a

import logging
from pathlib import Path
from dotenv import load_dotenv
from os import environ as env
import praw
from datetime import datetime
import pandas as pd
from joblib import dump
from typing import Optional, List

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


def get_save_directory(datasets_loc="datasets", reddit_dump_loc="reddit_dump"):
    current_directory = Path(__file__).resolve().parent
    save_directory = current_directory.parent / datasets_loc / reddit_dump_loc

    try:
        save_directory.mkdir()
    except FileExistsError:
        pass

    return save_directory


class ScrapeReddit:

    save_directory = get_save_directory()
    MAX_POSTS_PER_SUBREDDIT = 200
    subreddit_list = []

    @classmethod
    def config(
        cls,
        subreddits: Optional[List[str]] = [],
        max_posts_per_subreddit: Optional[int] = -1,
    ) -> bool:
        if subreddits:
            cls.subreddit_list = subreddits
            logger.info(
                f"Configured ScrapeReddit with new subreddits: {cls.subreddit_list}"
            )
        if max_posts_per_subreddit > 5:
            cls.MAX_POSTS_PER_SUBREDDIT = max_posts_per_subreddit
            logger.info(
                f"Configured ScrapeReddit with new max post limit of {cls.MAX_POSTS_PER_SUBREDDIT}"
            )

    @staticmethod
    def __praw_helper():
        load_dotenv()

        REDDIT_PERSONAL_USE_SCRIPT = env["REDDIT_PERSONAL_USE_SCRIPT"]
        REDDIT_SECRET = env["REDDIT_SECRET"]
        REDDIT_USERNAME = env["REDDIT_USERNAME"]
        REDDIT_PASSWORD = env["REDDIT_PASSWORD"]
        REDDIT_USERAGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.85 Safari/537.36"

        return praw.Reddit(
            client_id=REDDIT_PERSONAL_USE_SCRIPT,
            client_secret=REDDIT_SECRET,
            usernme=REDDIT_USERNAME,
            password=REDDIT_PASSWORD,
            user_agent=REDDIT_USERAGENT,
        )

    @classmethod
    def __retrieve_posts(cls):
        cls.save_directory

        reddit_instance = cls.__praw_helper()

        subred_list = []
        author_list = []
        link_flair_text_list = []
        num_comments_list = []
        score_list = []
        title_list = []
        upvote_ratio_list = []

        logger.info(f"Started scrapping at {datetime.now()}")

        for subred in cls.subreddit_list:
            logger.info(f"Starting to scrape {subred.upper()}")
            subreddit = reddit_instance.subreddit(subred)
            hot_post = subreddit.hot(limit=cls.MAX_POSTS_PER_SUBREDDIT)

            subreddit_post_count = 0
            for sub in hot_post:
                subred_list.append(subred)
                author_list.append(sub.author)
                link_flair_text_list.append(sub.link_flair_text)
                num_comments_list.append(sub.num_comments)
                score_list.append(sub.score)
                title_list.append(sub.title)
                upvote_ratio_list.append(sub.upvote_ratio)
                subreddit_post_count += 1

            logger.info(f"Scraped {subreddit_post_count} posts from {subred.upper()}")

        return {
            "Subreddit": subred_list,
            "Title": title_list,
            "Count_of_Comments": num_comments_list,
            "Upvote_Count": score_list,
            "Upvote_Ratio": upvote_ratio_list,
            "Flair": link_flair_text_list,
            "Author": author_list,
        }

    @classmethod
    def __posts_ETL(cls):
        posts_df = pd.DataFrame(cls.__retrieve_posts())
        posts_df.to_csv(cls.save_directory / "reddit_dataset.csv", index=False)

        dump(posts_df, cls.save_directory / "reddit_dataset.joblib")

    @classmethod
    def scrape(cls):
        cls.__posts_ETL()


def scrape():
    ScrapeReddit.config(
        [
            "wallstreetbets",
            "ocugen",
            "teslainvestorsclub",
            "SPACs",
            "worldnews",
            "singapore",
        ]
    )
    ScrapeReddit.scrape()


if __name__ == "__main__":
    scrape()