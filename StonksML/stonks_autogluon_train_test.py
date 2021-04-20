# https://auto.gluon.ai/stable/tutorials/text_prediction/beginner.html
# https://auto.gluon.ai/stable/install.html

import numpy as np
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
np.random.seed(123)

from autogluon.core.utils.loaders.load_pd import load
import joblib

# train_data = load(
#     "https://autogluon-text.s3-accelerate.amazonaws.com/glue/sst/train.parquet"
# )
test_data = load(
    "https://autogluon-text.s3-accelerate.amazonaws.com/glue/sst/dev.parquet"
)

# train_data.head(10)

# """training"""
# from autogluon.text import TextPredictor

# predictor = TextPredictor(label="label", eval_metric="acc", path="./ag_sst")
# predictor.fit(train_data, time_limit=60)

# """Evaluation"""
# test_score = predictor.evaluate(test_data, metrics=["acc", "f1"])
# print(test_score)

"""Intermediate Results"""
# predictor.results.tail(3)

from autogluon.text import TextPredictor

predictor = TextPredictor.load("./ag_sst/")


"""Direct prediction (Boolean)"""
# sentence1 = "it's a charming and often affecting journey."
# sentence2 = "It's slow, very, very, very slow."

sentence1 = "it's a charming and often affecting journey."
sentence2 = "It's slow, very, very, very slow."
predictions = predictor.predict({"sentence": [sentence1, sentence2]})
print('"Sentence":', sentence1, '"Predicted Sentiment":', predictions[0])
print('"Sentence":', sentence2, '"Predicted Sentiment":', predictions[1])

"""Probabilistic predictions"""
probs = predictor.predict_proba({"sentence": [sentence1, sentence2]})
print('"Sentence":', sentence1, '"Predicted Class-Probabilities":', probs[0])
print('"Sentence":', sentence2, '"Predicted Class-Probabilities":', probs[1])


"""Save and load"""
loaded_predictor = TextPredictor.load("ag_sst")  # load automatically saved predictor"
loaded_predictor.predict_proba({"sentence": [sentence1, sentence2]})

"""Saving to custom location"""
loaded_predictor.save("my_saved_dir")
loaded_predictor2 = TextPredictor.load("my_saved_dir")
loaded_predictor2.predict_proba({"sentence": [sentence1, sentence2]})

"""Extract embedding"""
embeddings = predictor.extract_embedding(test_data)
print(embeddings)

from sklearn.manifold import TSNE

X_embedded = TSNE(n_components=2, random_state=123).fit_transform(embeddings)
for val, color in [(0, "red"), (1, "blue")]:
    idx = (test_data["label"].to_numpy() == val).nonzero()
    plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], c=color, label=f"label={val}")
plt.legend(loc="best")
