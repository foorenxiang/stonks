import os
from pathlib import Path
import shutil


def clean_model_dumps():
    model_dumps_dir = Path(__file__).resolve().parent / "autots_model_dumps"
    model_categories = ["ACCURATE", "FAST", "GPU", "DEFAULT"]

    folders_to_delete = []

    for category in model_categories:
        folders_to_delete_for_category = [
            dir for dir in model_dumps_dir.glob(f"*{category}*") if dir.is_dir()
        ]

        folders_to_delete = filter(
            lambda dir: not (
                dir == (max(folders_to_delete_for_category, key=os.path.getmtime))
            ),
            folders_to_delete_for_category,
        )

    [shutil.rmtree(folder, ignore_errors=True) for folder in folders_to_delete]


if __name__ == "__main__":
    clean_model_dumps()
