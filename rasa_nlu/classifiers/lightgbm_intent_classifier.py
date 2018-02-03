import io
import logging
import os

import lightgbm as lgb
import numpy as np
from future.utils import PY3
from sklearn.preprocessing import LabelEncoder

from rasa_nlu.classifiers.utils import transform_labels_str2num, transform_labels_num2str
from rasa_nlu.components import Component

logger = logging.getLogger(__name__)

INTENT_RANKING_LENGTH = 10


class LightGBMIntentClassifier(Component):
    """Intent classifier using the lightgbm framework"""

    name = "intent_classifier_lightgbm"

    provides = ["intent"]

    requires = ["text_features"]

    def __init__(self, clf=None, le=None):
        # type: (lgb.LGBMClassifier, LabelEncoder)->None
        self.clf = clf

        if le is not None:
            self.le = le
        else:
            self.le = LabelEncoder()

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["numpy", "lightgbm"]

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData, RasaNLUConfig, **Any) -> None
        """Train the intent classifier on a data set.
        """

        labels = [e.get("intent") for e in training_data.intent_examples]

        if len(set(labels)) < 2:
            logger.warn("Can not train an intent classifier. Need at least 2 different classes. " +
                        "Skipping training of intent classifier.")
        else:
            y = transform_labels_str2num(self.le, labels)
            X = np.stack([example.get("text_features") for example in training_data.intent_examples])

            self.clf = lgb.LGBMClassifier(n_estimators=20, num_leaves=63)
            self.clf.fit(X, y,verbose=True)

    def predict_prob(self, X):
        # type: (np.ndarray) -> np.ndarray
        """Given a bow vector of an input text, predict the intent label. Returns probabilities for all labels.

        :param X: bow of input text
        :return: vector of probabilities containing one entry for each label"""

        return self.clf.predict_proba(X)

    def predict(self, X):
        # type: (np.ndarray) -> Tuple[np.ndarray, np.ndarray]
        """Given a bow vector of an input text, predict most probable label. Returns only the most likely label.

        :param X: bow of input text
        :return: tuple of first, the most probable label and second, its probability"""

        import numpy as np

        pred_result = self.predict_prob(X)
        # sort the probabilities retrieving the indices of the elements in sorted order
        sorted_indices = np.fliplr(np.argsort(pred_result, axis=1))
        return sorted_indices, pred_result[:, sorted_indices]

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None
        """Returns the most likely intent and its probability for the input text."""

        if not self.clf:
            # component is either not trained or didn't receive enough training data
            intent = None
            intent_ranking = []
        else:
            X = message.get("text_features").reshape(1, -1)
            intent_ids, probabilities = self.predict(X)
            intents = transform_labels_num2str(self.le, intent_ids)
            # `predict` returns a matrix as it is supposed
            # to work for multiple examples as well, hence we need to flatten
            intents, probabilities = intents.flatten(), probabilities.flatten()

            if intents.size > 0 and probabilities.size > 0:
                ranking = list(zip(list(intents), list(probabilities)))[:INTENT_RANKING_LENGTH]
                intent = {"name": intents[0], "confidence": probabilities[0]}
                intent_ranking = [{"name": intent_name, "confidence": score} for intent_name, score in ranking]
            else:
                intent = {"name": None, "confidence": 0.0}
                intent_ranking = []

        message.set("intent", intent, add_to_output=True)
        message.set("intent_ranking", intent_ranking, add_to_output=True)

    @classmethod
    def load(cls, model_dir=None, model_metadata=None, cached_component=None, **kwargs):
        # type: (Text, Metadata, Optional[Component], **Any) -> LightGBMIntentClassifier
        import cloudpickle

        if model_dir and model_metadata.get("intent_classifier_lightgbm"):
            classifier_file = os.path.join(model_dir, model_metadata.get("intent_classifier_lightgbm"))
            with io.open(classifier_file, 'rb') as f:  # pragma: no test
                if PY3:
                    return cloudpickle.load(f, encoding="latin-1")
                else:
                    return cloudpickle.load(f)
        else:
            return LightGBMIntentClassifier()

    def persist(self, model_dir):
        # type: (Text) -> Dict[Text, Any]
        """Persist this model into the passed directory. Returns the metadata necessary to load the model again."""

        import cloudpickle

        classifier_file = os.path.join(model_dir, "intent_classifier.pkl")
        with io.open(classifier_file, 'wb') as f:
            cloudpickle.dump(self, f)

        return {
            "intent_classifier_lightgbm": "intent_classifier.pkl"
        }
