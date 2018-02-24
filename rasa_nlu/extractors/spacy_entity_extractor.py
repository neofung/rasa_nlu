from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals, print_function

import logging
import os
import random
import typing
from typing import Any
from typing import Dict
from typing import List
from typing import Text

from spacy.pipeline import EntityRecognizer
from spacy.util import minibatch
from tqdm import tqdm

from rasa_nlu.extractors import EntityExtractor
from rasa_nlu.training_data import Message, TrainingData

DISABLE_ORIGINAL_SPACY_NER = 'disable_original_spacy_ner'

if typing.TYPE_CHECKING:
    from spacy.tokens.doc import Doc

logger = logging.getLogger(__name__)


class SpacyEntityExtractor(EntityExtractor):
    name = "ner_spacy"

    provides = ["entities"]

    requires = ["spacy_doc", "spacy_nlp"]

    def __init__(self, ner_pipes=None):
        # type: (Dict(EntityExtractor)) -> None
        self.ner_pipes = ner_pipes

    def provide_context(self):
        # type: () -> Dict[Text, Any]

        return {DISABLE_ORIGINAL_SPACY_NER: True}

    @classmethod
    def load(cls, model_dir=None, model_metadata=None, cached_component=None, **kwargs):
        ner_pipes = {}
        spacy_nlp = kwargs['spacy_nlp']
        path = model_dir + '/ner.model'
        os.listdir(path)
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if os.path.isdir(file_path):
                ner = EntityRecognizer(spacy_nlp.vocab).from_disk(file_path, vocab=False)
                ner_pipes[file] = ner

        extractor = SpacyEntityExtractor(ner_pipes)

        return extractor

    def persist(self, model_dir):
        path = model_dir + '/ner.model'
        os.mkdir(path)
        for key in self.ner_pipes.keys():
            ner = self.ner_pipes[key]
            file_path = os.path.join(path, key)
            ner.to_disk(file_path, vocab=False)

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData, RasaNLUConfig, **Any) -> Dict[Text, Any]

        spacy_nlp = kwargs['spacy_nlp']
        self.ner_pipes = {}

        for intent in training_data.intents:
            training_ner_data = []

            label_set = set()

            for example in training_data.entity_examples:
                if example.get('intent') == intent:
                    entities = [(t['start'], t['end'], t['entity']) for t in example.get('entities')]
                    for _, _, entity in entities:
                        label_set.add(entity)
                    training_ner_data.append((example.text, {'entities': entities}))

            if not training_ner_data:
                return {}

            # get the ner pipe
            ner = spacy_nlp.create_pipe('ner')
            for label in label_set:
                ner.add_label(label)

            spacy_nlp.replace_pipe('ner', ner)
            self.__train_ner(config.get('ner_spacy'), spacy_nlp, intent, training_ner_data)
            self.ner_pipes[intent] = ner

        return {}

    @staticmethod
    def __train_ner(ner_config, nlp, intent, training_ner_data):
        batch_size, epochs, gpu_device = SpacyEntityExtractor.get_train_param(ner_config)
        epochs = 1 if not training_ner_data else epochs

        # get names of other pipes to disable them during training
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
        with nlp.disable_pipes(*other_pipes):  # only train NER
            optimizer = nlp.begin_training(device=gpu_device)
            for it in range(epochs):
                random.shuffle(training_ner_data)
                losses = {}

                batches = minibatch(training_ner_data, size=batch_size)
                progress = tqdm(batches, total=len(training_ner_data) / batch_size, desc='intent: %s' % intent)
                for batch in progress:
                    texts, annotations = zip(*batch)
                    nlp.update(texts, annotations, sgd=optimizer, drop=0.35,
                               losses=losses)
                    progress.set_description_str('intent: %s, epoch: %d/%d, loss: %s'
                                                 % (intent, it + 1, epochs, str(losses)))

    @staticmethod
    def get_train_param(ner_config):
        if not ner_config:
            logger.warning('ner_config is None')
            epochs = 16
            batch_size = 10
            gpu_device = -1
        else:
            epochs = ner_config.get('epochs', 16)
            batch_size = ner_config.get('batch_size', 10)
            gpu_device = ner_config.get('gpu_device', -1)
        return batch_size, epochs, gpu_device

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        intent = message.get('intent')
        doc = message.get('spacy_doc')
        extracted = []
        if intent and intent['name'] and intent['name'] in self.ner_pipes:
            ner = self.ner_pipes[intent['name']]
            ner(doc)

            extracted = self.add_extractor_name(self.extract_entities(message.get("spacy_doc")))
        message.set("entities", message.get("entities", []) + extracted, add_to_output=True)

    def extract_entities(self, doc):
        # type: (Doc) -> List[Dict[Text, Any]]

        entities = [
            {
                "entity": ent.label_,
                "value": ent.text,
                "start": ent.start_char,
                "end": ent.end_char
            }
            for ent in doc.ents]
        return entities
