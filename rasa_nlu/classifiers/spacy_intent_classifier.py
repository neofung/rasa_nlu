import random

from spacy.util import minibatch, compounding
from spacy.pipeline import TextCategorizer
from tqdm import tqdm

from rasa_nlu.components import Component
from rasa_nlu.training_data import TrainingData, Message


class SpacyIntentClassifier(Component):
    name = 'intent_classifier_spacy'

    provides = ["intent"]

    requires = ['spacy_nlp']

    def __init__(self, classifier=None):
        # type: (TextCategorizer) -> None
        self.classifier = classifier

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData, RasaNLUConfig, **Any) -> Dict[Text, Any]
        assert isinstance(training_data, TrainingData)

        spacy_config = config.get('intent_classifier_spacy')

        batch_size = spacy_config.get('batch_size', 32)
        epochs = spacy_config.get('epochs', 10)
        dropout = spacy_config.get('dropout', 0.25)
        gpu_device = spacy_config.get('gpu_device', -1)

        def foo(intent):
            cats = {}
            for key in training_data.intents:
                cats[key] = key == intent
            return cats

        spacy_nlp = kwargs['spacy_nlp']

        if 'textcat' not in spacy_nlp.pipe_names:
            classifier = spacy_nlp.create_pipe('textcat')
            spacy_nlp.add_pipe(classifier, last=True)
        # otherwise, get it, so we can add labels to it
        else:
            classifier = spacy_nlp.get_pipe('textcat')

        classifier.cfg['rasa_updated'] = True

        for label in training_data.intents:
            classifier.add_label(label)

        train_texts = [example.text for example in training_data.intent_examples]
        train_cats = [foo(example.get('intent')) for example in training_data.intent_examples]
        train_data = list(zip(train_texts,
                              [{'cats': cats} for cats in train_cats]))

        other_pipes = [pipe for pipe in spacy_nlp.pipe_names if pipe != classifier.name]

        with spacy_nlp.disable_pipes(*other_pipes):  # only train textcat
            optimizer = spacy_nlp.begin_training(device=gpu_device)
            for it in range(epochs):
                losses = {}
                random.shuffle(train_data)
                # batch up the examples using spaCy's minibatch
                batches = minibatch(train_data, size=batch_size)
                progress = tqdm(batches, total=len(train_data) / batch_size)
                for batch in progress:
                    texts, annotations = zip(*batch)
                    spacy_nlp.update(texts, annotations, sgd=optimizer, drop=dropout,
                                     losses=losses)
                    progress.set_description_str('epoch %d/%d, loss: %s' % (it + 1, epochs, str(losses)))

        self.classifier = classifier
        return {'spacy_nlp': spacy_nlp}

    @classmethod
    def required_packages(cls):
        return ['spacy']

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        assert isinstance(message, Message)

        spacy_doc = message.get('spacy_doc', None)
        intents = spacy_doc.cats.items() if spacy_doc else None
        if intents:
            ranking = sorted(intents, key=lambda x: -x[1])
            intent = {"name": ranking[0][0], "confidence": ranking[0][1],
                      "classifier": self.name}
            intent_ranking = [
                {"name": intent_name, "confidence": score, "classifier": self.name} for
                intent_name, score in ranking]
        else:
            intent = {"name": None, "confidence": 0.0, "classifier": self.name}
            intent_ranking = []

        message.set("intent", intent, add_to_output=True)
        message.set("intent_ranking", intent_ranking, add_to_output=True)

    @classmethod
    def load(cls, model_dir=None, model_metadata=None, cached_component=None, **kwargs):
        spacy_nlp = kwargs['spacy_nlp']
        path = model_dir + '/intent_classifier.model'
        textcat = TextCategorizer(spacy_nlp.vocab).from_disk(path, vocab=False)

        classifier = SpacyIntentClassifier(textcat)

        spacy_nlp.add_pipe(classifier.classifier, last=True)

        return classifier

    def persist(self, model_dir):
        path = model_dir + '/intent_classifier.model'
        self.classifier.to_disk(path, vocab=False)
