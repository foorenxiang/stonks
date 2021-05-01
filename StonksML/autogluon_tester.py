import logging
import joblib
from pathlib import Path
from autogluon.text import TextPredictor

current_directory = Path(__file__).resolve().parent
logging_directory = current_directory / "autogluon_logs"

try:
    logging_directory.mkdir()
except FileExistsError:
    pass

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler(
            filename=logging_directory / "autogluon_model_accuracy_test.log",
            mode="w",
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger()


def autogluon_model_test():
    model_save_path = Path(__file__).resolve().parent / "autogluon_model"
    predictor = TextPredictor.load(model_save_path)

    # test_data = pd.read_csv(Path(__file__).resolve().parent / "test_data.csv").iloc[
    #     :5000, 1:
    # ]
    test_data_path = (
        current_directory.parent
        / "datasets"
        / "preprocessed"
        / "full_preprocessed_reddit_twitter_dataset.joblib"
    )
    test_data = joblib.load(test_data_path)[["sentence", "label"]]

    test_sentences = test_data["sentence"].tolist()
    labels = test_data["label"].tolist()
    predictions = predictor.predict({"sentence": test_sentences})

    num_corrects = 0

    def determine_score(prediction, label):
        nonlocal num_corrects
        score_calculation = prediction - label
        if score_calculation == 0:
            num_corrects += 1
            return "🤩ok"
        elif score_calculation == -1:
            return "💩FP"
        return "💀FN"

    for prediction, label, sentence in zip(predictions, labels, test_sentences):
        score = determine_score(prediction, label)
        output = f"{score} {sentence}"
        logger.info(output)

    total_trials = len(test_sentences)
    percentage_correct = num_corrects / total_trials * 100

    logger.info(f"Score: {num_corrects}/{total_trials} ({percentage_correct}%)")
    logger.info("\n\n\n")

    sentences = test_sentences + [
        "it's a charming and often affecting journey.",
        "It's slow, very, very, very slow.",
        "happy",
        "sad",
        "oh no",
        ":O",
        "omg",
        "diamond hands",
        "paper hands",
        "lit stocks",
        "to the moon",
        "gamestop",
        "losses",
        "roaring kitty",
        "yolo",
        "bagholder",
        "tendies",
    ]
    predictions = predictor.predict({"sentence": sentences[-50:]})

    [
        logger.info(f'{"🤩" if not not prediction else "😭/😐"} {sentences[idx]}')
        for idx, prediction in enumerate(predictions)
    ]


if __name__ == "__main__":
    autogluon_model_test()
