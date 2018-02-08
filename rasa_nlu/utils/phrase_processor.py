import logging
from copy import deepcopy

from rasa_nlu.components import Component
from rasa_nlu.training_data import TrainingData, Message

logger = logging.getLogger(__name__)


class PhraseProcessor(Component):
    name = 'phrase_processor'

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData, RasaNLUConfig, **Any) -> Dict[Text, Any]
        assert isinstance(training_data, TrainingData)
        phrase_lists = training_data.phrase_lists
        new_examples = self.__create_examples(phrase_lists, training_data.training_examples)

        training_data.training_examples.extend(new_examples)
        training_data.update_lazyproperty()

    def __create_examples(self, phrase_lists, examples):
        new_examples = []
        for phrase_list in phrase_lists:
            for example in examples:
                assert isinstance(example, Message)
                entities = example.get('entities', [])
                for i, origin_entity in enumerate(entities):
                    if origin_entity['entity'] in phrase_list['entities']:
                        for value in phrase_list['values']:
                            if origin_entity['value'] != value:
                                new_example = self.__fork_example(i, example, value)

                                new_examples.append(new_example)
        return new_examples

    def __fork_example(self, i, origin_example, value):
        new_example = deepcopy(origin_example)
        new_entity = new_example.get('entities')[i]
        new_example.text = new_example.text[:new_entity['start']] + value + new_example.text[new_entity['end']:]
        diff = new_entity['end'] - new_entity['start'] - len(value)
        new_entity['value'] = value
        if diff:
            self.__update_index(diff, new_entity, new_example)
        return new_example

    @staticmethod
    def __update_index(diff, new_entity, example):
        new_entity['end'] -= diff
        entities = example.get('entities')
        for entity in entities:
            if entity['start'] > new_entity['start']:
                entity['start'] -= diff
                entity['end'] -= diff
