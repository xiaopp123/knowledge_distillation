# -*- coding: utf-8 -*-


import os
import random


class InputExample(object):
    def __init__(self, guid, text, label=None):
        self.guid = guid
        self.text = text
        self.label = label


class Processor(object):
    def get_train_examples(self, data_dir):
        return self._create_examples(
            os.path.join(data_dir, 'train.txt'), 'train')

    def get_test_examples(self, data_dir):
        return self._create_examples(
            os.path.join(data_dir, 'test.txt'), 'test')

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            os.path.join(data_dir, 'dev.txt'), 'dev')

    def get_labels(self):
        return ['0', '1']

    def _create_examples(self, data_path, set_type):
        examples = []
        with open(data_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                label, text = line.strip().split('\t', 1)
                guid = "{0}-{1}-{2}".format(set_type, label, i)
                examples.append(InputExample(guid=guid, text=text, label=label))
        random.shuffle(examples)
        return examples


def _test():
    process = Processor()
    train_examples = process.get_train_examples(data_dir='../data/hotel')
    for example in train_examples:
        print(example.guid)
        print(example.text)
        print(example.label)
        break


if __name__ == '__main__':
    _test()
