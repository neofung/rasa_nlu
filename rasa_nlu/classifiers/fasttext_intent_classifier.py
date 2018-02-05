import io
import logging
import os

import fastText
from future.utils import PY3
from sklearn.preprocessing import LabelEncoder

from rasa_nlu.classifiers.utils import transform_labels_str2num, transform_labels_num2str
from rasa_nlu.components import Component

logger = logging.getLogger(__name__)

INTENT_RANKING_LENGTH = 10


class FastTextIntentClassifier(Component):
    """Intent classifier using the lightgbm framework"""

    name = "intent_classifier_fasttext"

    provides = ["intent"]

    requires = ["tokens"]

    __LABEL_PREFIX = '__label__'

    def __init__(self, clf=None, le=None):
        # type: (fastText.model, LabelEncoder)->None
        self.clf = clf

        if le is not None:
            self.le = le
        else:
            self.le = LabelEncoder()

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["numpy", "fastText"]

    def _tokens_of_message(self, message):
        return [token.text for token in message.get("tokens", [])]

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData, RasaNLUConfig, **Any) -> None
        """Train the intent classifier on a data set.
        """

        tokens_list = []
        label_list = []
        for example in training_data.intent_examples:
            tokens_list.append(' '.join(self._tokens_of_message(example)))
            label_list.append(example.get('intent'))

        label_list = transform_labels_str2num(self.le, label_list)

        num_class = len(set(label_list))
        if num_class < 2:
            logger.warn("Can not train an intent classifier. Need at least 2 different classes. " +
                        "Skipping training of intent classifier.")
        else:
            train_data_intermediate = os.path.dirname(os.path.abspath(__file__)) + '/temp.txt'
            with open(train_data_intermediate, 'w') as fout:
                for label, content in zip(label_list, tokens_list):
                    fout.write('%s%s\t%s\n' % (FastTextIntentClassifier.__LABEL_PREFIX, label, content))

            self.clf = fastText.train_supervised(
                input=train_data_intermediate, epoch=25, lr=1.0, wordNgrams=3, verbose=2, minCount=1,
                loss="hs"
            )

        def print_results(N, p, r):
            logger.info("N\t" + str(N))
            logger.info("P@{}\t{:.3f}".format(1, p))
            logger.info("R@{}\t{:.3f}".format(1, r))

        print_results(*self.clf.test(train_data_intermediate))

    @classmethod
    def load(cls, model_dir=None, model_metadata=None, cached_component=None, **kwargs):
        # type: (Text, Metadata, Optional[Component], **Any) -> FastTextIntentClassifier
        import cloudpickle

        if model_dir and model_metadata.get("intent_classifier_fasttext"):
            classifier_file = os.path.join(model_dir, model_metadata.get("intent_classifier_fasttext"))
            clf = fastText.load_model(classifier_file)

            label_encoder_file = os.path.join(model_dir, model_metadata.get("intent_label_encoder"))
            with io.open(label_encoder_file, 'rb') as f:  # pragma: no test
                if PY3:
                    le = cloudpickle.load(f, encoding="latin-1")
                else:
                    le = cloudpickle.load(f)
            return FastTextIntentClassifier(clf, le)
        else:
            return FastTextIntentClassifier()

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None
        """Returns the most likely intent and its probability for the input text."""

        if not self.clf:
            # component is either not trained or didn't receive enough training data
            intent = None
            intent_ranking = []
        else:
            token_strs = ' '.join(self._tokens_of_message(message))

            labels, probabilities = self.clf.predict(token_strs, INTENT_RANKING_LENGTH)
            labels = [int(item[len(FastTextIntentClassifier.__LABEL_PREFIX):]) for item in labels]
            intents = transform_labels_num2str(self.le, labels)

            # `predict` returns a matrix as it is supposed
            # to work for multiple examples as well, hence we need to flatten
            probabilities = probabilities.flatten()

            if intents.size > 0 and probabilities.size > 0:
                ranking = list(zip(list(intents), list(probabilities)))[:INTENT_RANKING_LENGTH]
                intent = {"name": intents[0], "confidence": probabilities[0]}
                intent_ranking = [{"name": intent_name, "confidence": score} for intent_name, score in ranking]
            else:
                intent = {"name": None, "confidence": 0.0}
                intent_ranking = []

        message.set("intent", intent, add_to_output=True)
        message.set("intent_ranking", intent_ranking, add_to_output=True)

    def persist(self, model_dir):
        # type: (Text) -> Dict[Text, Any]
        """Persist this model into the passed directory. Returns the metadata necessary to load the model again."""

        import cloudpickle

        classifier_file = os.path.join(model_dir, "intent_classifier.bin")
        self.clf.save_model(classifier_file)
        intent_label_encoder = os.path.join(model_dir, "intent_label_encoder.pkl")
        with io.open(intent_label_encoder, 'wb') as f:
            cloudpickle.dump(self.le, f)

        return {
            "intent_classifier_fasttext": "intent_classifier.bin",
            "intent_label_encoder": "intent_label_encoder.pkl"
        }
