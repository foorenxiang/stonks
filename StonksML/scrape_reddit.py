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
    MAX_POSTS_PER_SUBREDDIT = 100
    subreddit_list = []
    flair_filters = []
    __posts = {}

    @classmethod
    def config(
        cls,
        subreddits: Optional[List[str]] = [],
        flair_filters: Optional[List[str]] = [],
        max_posts_per_subreddit: Optional[int] = -1,
    ) -> bool:
        if subreddits:
            cls.subreddit_list = subreddits
            logger.info(
                f"Configured ScrapeReddit with subreddits: {cls.subreddit_list}"
            )
        if flair_filters:
            cls.flair_filters = flair_filters
            logger.info(
                f"Configured ScrapeReddit with flair filters: {cls.flair_filters}"
            )
        if max_posts_per_subreddit > 5:
            cls.MAX_POSTS_PER_SUBREDDIT = max_posts_per_subreddit
            logger.info(
                f"Configured ScrapeReddit with max post limit of {cls.MAX_POSTS_PER_SUBREDDIT}"
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
    def is_valid_flair(cls, flair):
        if not flair:
            return True
        for filter in cls.flair_filters:
            if filter.upper() in flair.upper():
                return False
        return True

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
            hot_posts = subreddit.hot(limit=cls.MAX_POSTS_PER_SUBREDDIT * 2)

            subreddit_post_count = 0

            for post in hot_posts:
                sub_flair = post.link_flair_text

                if cls.is_valid_flair(sub_flair):
                    subred_list.append(subred)
                    author_list.append(post.author)
                    link_flair_text_list.append(post.link_flair_text)
                    num_comments_list.append(post.num_comments)
                    score_list.append(post.score)
                    title_list.append(post.title)
                    upvote_ratio_list.append(post.upvote_ratio)
                    subreddit_post_count += 1

                if subreddit_post_count == cls.MAX_POSTS_PER_SUBREDDIT:
                    break

            logger.info(f"Scraped {subreddit_post_count} posts from {subred.upper()}")

        cls.__posts = {
            "Subreddit": subred_list,
            "Title": title_list,
            "Count_of_Comments": num_comments_list,
            "Upvote_Count": score_list,
            "Upvote_Ratio": upvote_ratio_list,
            "Flair": link_flair_text_list,
            "Author": author_list,
        }

    @staticmethod
    def __flairs(posts_df):
        try:
            assert not posts_df.empty
        except AssertionError:
            logger.warn("No posts to check!")
            return
        except AttributeError:
            logger.error("flairs expect a dataframe!")
            raise
        flairs = posts_df["Flair"].unique()
        logger.info("Flairs:")
        [logger.info(flair) for flair in flairs]

    @classmethod
    def __posts_ETL(cls):
        try:
            assert cls.__posts
        except AssertionError:
            logger.error("No posts scraped yet!")
            return

        posts_df = pd.DataFrame(cls.__posts)
        cls.__flairs(posts_df)
        posts_df.to_csv(cls.save_directory / "reddit_dataset.csv", index=False)

        dump(posts_df, cls.save_directory / "reddit_dataset.joblib")

    @classmethod
    def scrape(cls):
        cls.__retrieve_posts()
        cls.__posts_ETL()


def scrape():
    ScrapeReddit.config(
        [
            "wallstreetbets",
            "ocugen",
            "teslainvestorsclub",
            "SPACs",
        ],
        [
            "Weekend",
            "Analysis",
            "Daily",
            "Discussion",
            "Thread",
            "Mods",
        ],
        50,
    )
    ScrapeReddit.scrape()


if __name__ == "__main__":
    scrape()