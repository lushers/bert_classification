#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2019 lushers, Inc. All Rights Reserved
#
########################################################################
"""
File: bert_utils.py
Author: lushers
Date: 2019/06/21 10:57:31
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import codecs
from bert import modeling, optimization, tokenization
import tensorflow as tf


class InputExample(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        lists = []
        with codecs.open(input_file, "r", "utf-8") as fd:
            for line in fd:
                lis = line.strip().split('\t|@@@|\t')
                if len(lis) != 3:
                    continue
                lis[1] = lis[1].replace(u'\n', u'').lstrip(u'。').replace(u'。。', u'')
                lis[2] = lis[2].replace(u'\n', u'').lstrip(u'。').replace(u'。。', u'')
                lists.append(lis)
        return lists

class MineABProcessor(DataProcessor):
    '''mine data'''
    def __init__(self):
        self.labels = self.get_labels()

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'train.shuf')), 'train')

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.shuf")), 'dev')

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), 'test')

    def get_labels(self):
        """Gets the list of labels for this data set."""
        # 设置所有labels 标签
        labels = []
        with codecs.open('labels.txt', 'r', 'utf-8') as fd:
            for line in fd:
                labels.append(line.strip())
        return labels

    def _create_examples(self, lines, set_type):
        '''create examples for training and dev'''
        examples = []
        for (i, line) in enumerate(lines):
            guid = '%s-%s' % (set_type, i)
            # title + content
            try:
                text_a = tokenization.convert_to_unicode(line[1])
                text_b = tokenization.convert_to_unicode(line[2])
                if set_type == "test":
                    label = u"体育"   # default
                else:
                    label = tokenization.convert_to_unicode(line[0])
                    if label not in self.labels:
                        continue
            except Exception as ex:
                continue
            examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MineProcessor(DataProcessor):
    '''mine data'''
    def __init__(self):
        self.labels = self.get_labels()

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "ori/train_shuf.tsv")), 'train')

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "ori/dev_shuf.tsv")), 'dev')

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), 'test')

    def get_labels(self):
        """Gets the list of labels for this data set."""
        # 设置所有labels 标签
        labels = []
        with codecs.open('labels.txt', 'r', 'utf-8') as fd:
            for line in fd:
                labels.append(line.strip())
        return labels

    def _create_examples(self, lines, set_type):
        '''create examples for training and dev'''
        examples = []
        for (i, line) in enumerate(lines):
            guid = '%s-%s' % (set_type, i)
            # title + content
            try:
                text_a = tokenization.convert_to_unicode(line[1] + line[2])
                if set_type == "test":
                    label = u"美食"
                else:
                    label = tokenization.convert_to_unicode(line[0])
                    if label not in self.labels:
                        continue
            except Exception as ex:
                continue
            examples.append(
                    InputExample(guid=guid, text_a=text_a, label=label))
        return examples

def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_id = label_map[example.label]

    if ex_index < 3:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

    feature = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label_id=label_id)
    return feature

def file_based_convert_examples_to_features(
         examples, label_list, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        feature = convert_single_example(ex_index, example, label_list,
                max_seq_length, tokenizer)
        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f
        def create_float_feature(values):
            f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
            return f
        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature([feature.label_id])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()

def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
            "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
            "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "label_ids": tf.FixedLenFeature([], tf.int64)
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=3700)
        d = d.apply(
                tf.contrib.data.map_and_batch(
                    lambda record: _decode_record(record, name_to_features),
                    batch_size=batch_size,
                    drop_remainder=drop_remainder))
        return d
    return input_fn

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    # 将两个句子相加，如果长度大于max_length 就pop 直到小于 max_length
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
