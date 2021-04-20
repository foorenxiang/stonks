import logging
from autogluon.text import TextPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

predictor = TextPredictor.load("./ag_sst/")

sentences = [
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
predictions = predictor.predict({"sentence": sentences})

[
    logger.info(
        f'"Sentence": {sentences[idx]}      "Predicted Sentiment": {"positive" if not not prediction else "negative"}'
    )
    for idx, prediction in enumerate(predictions)
]

predictions_proba = predictor.predict_proba({"sentence": sentences})

[
    logger.info(
        f'"Sentence": {sentences[idx]}      "Predicted Sentiment Probability": {(prediction - 0.5) * 2}'
        # '"Sentence":',
        # sentences[idx],
        # '"Predicted Sentiment Probability":',
        # (prediction - 0.5) * 2,
    )
    for idx, prediction in enumerate(predictions_proba.iloc[:, 1])
]
