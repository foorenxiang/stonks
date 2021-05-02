import logging
import joblib
from pathlib import Path
from utils import paths_catalog
from autogluon_train_test import NUM_ROWS_TO_EXTRACT as HEAD_ROWS_USED_IN_TRAINING
from autogluon.text import TextPredictor

CURRENT_DIRECTORY = Path(__file__).resolve().parent
LOGGING_DIRECTORY = paths_catalog.AUTOGLUON_LOGS
LOGFILE_NAME = "autogluon_model_accuracy_test.log"


def autogluon_model_test():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())

    try:
        LOGGING_DIRECTORY.mkdir()
    except FileExistsError:
        logger.info(f"{LOGGING_DIRECTORY} exists, using it")

    logger.addHandler(
        logging.FileHandler(
            filename=LOGGING_DIRECTORY / LOGFILE_NAME,
            mode="w",
        )
    )

    FULL_DATASET = "full_preprocessed_reddit_twitter_dataset.joblib"
    predictor = TextPredictor.load(paths_catalog.AUTOGLUON_MODEL)
    FULL_DATASET_PATH = paths_catalog.PREPROCESSED_DATASETS / FULL_DATASET
    full_dataset = joblib.load(FULL_DATASET_PATH)

    NUM_ROWS_IN_FULL_DATASET = full_dataset.shape[0]
    TAIL_ROWS_NOT_USED_IN_TRAINING = (
        NUM_ROWS_IN_FULL_DATASET - HEAD_ROWS_USED_IN_TRAINING
    )
    DEFAULT_SAMPLE_SIZE = 10000
    MINIMUM_PREFERRED_SAMPLE_SIZE = 2000
    sample_tail_size = TAIL_ROWS_NOT_USED_IN_TRAINING
    if sample_tail_size > DEFAULT_SAMPLE_SIZE:
        sample_tail_size = DEFAULT_SAMPLE_SIZE
    elif sample_tail_size < MINIMUM_PREFERRED_SAMPLE_SIZE:
        sample_tail_size = DEFAULT_SAMPLE_SIZE
        logger.warning(
            f"Only {TAIL_ROWS_NOT_USED_IN_TRAINING} rows were not used in training, evaluating with tail {DEFAULT_SAMPLE_SIZE} rows instead"
        )

    test_data = full_dataset[-sample_tail_size:][["sentence", "label"]]

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
    BASELINE_ACCURACY = 75

    logger.info(f"Score: {num_corrects}/{total_trials} ({percentage_correct}%)")
    if percentage_correct < BASELINE_ACCURACY:
        logger.warning(
            f"Sentiment analysis model has accuracy of less than {BASELINE_ACCURACY}%, consider retraining it!!"
        )


if __name__ == "__main__":
    autogluon_model_test()
