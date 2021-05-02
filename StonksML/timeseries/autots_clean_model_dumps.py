import os
import logging
from pathlib import Path
import shutil

import sys
from from_root import from_root

sys.path.append(str(from_root(".")))
from utils import paths_catalog

CURRENT_DIRECTORY = Path(__file__).resolve().parent
MODEL_DUMPS_DIRECTORY = paths_catalog.AUTOTS_MODEL_DUMPS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_model_dumps():
    model_categories = ["ACCURATE", "FAST", "GPU", "DEFAULT"]

    folders_to_delete = []

    for category in model_categories:
        folders_to_delete_for_category = [
            dir for dir in MODEL_DUMPS_DIRECTORY.glob(f"*{category}*") if dir.is_dir()
        ]

        folders_to_delete = filter(
            lambda dir: not (
                dir == (max(folders_to_delete_for_category, key=os.path.getmtime))
            ),
            folders_to_delete_for_category,
        )

    for folder in folders_to_delete:
        logger.info(f"Purging {folder}")
        shutil.rmtree(folder, ignore_errors=True)


if __name__ == "__main__":
    clean_model_dumps()
