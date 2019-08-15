#!/usr/bin/env python
#coding:utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
sys.path.append("..")

import os
import tensorflow as tf
from bert import tokenization
import numpy as np
import json
import codecs
from eva_config import Config

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
gpu_options = tf.GPUOptions(allow_growth=True)

class BertSimple():
    '''
        BertModel for sentence embedding
    '''
    def __init__(self):
        self.config = Config()
        self.tokenizer = tokenization.FullTokenizer(self.config.vocab_file, True)
        self.label_map = self._get_label()
        self.graph = self.load_graph()
        self.input_ids = self.graph.get_tensor_by_name('prefix/input_ids:0')
        self.input_mask = self.graph.get_tensor_by_name('prefix/input_mask:0')
        self.input_type_ids = self.graph.get_tensor_by_name('prefix/input_type_ids:0')
        self.y_out = self.graph.get_tensor_by_name('prefix/pooling/predict:0')
        self.y_prb = self.graph.get_tensor_by_name('prefix/pooling/probabilities:0')

        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=self.graph)
        #self.sess = tf.Session(graph=self.graph)

    def _get_label(self):
        """
            get label_map
        """
        labels = []
        with codecs.open('labels.txt', 'r', 'utf-8') as fd:
            for line in fd:
                labels.append(line.strip())
        label_map = {}
        for (i, label) in enumerate(labels):
            label_map[i] = label
        return label_map

    def load_graph(self):
        '''
            load graph
        '''
        with tf.gfile.GFile(self.config.model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(
                    graph_def,
                    input_map=None,
                    return_elements=None,
                    name='prefix',
                    op_dict=None,
                    producer_op_list=None)

        return graph

    def _convert_single_sentence(self, title, content=None, max_seq_length=256):
        '''
            convert a sentence to a numpy array
            text_a: CLS [seq] SEP
        '''
        tokens_a = self.tokenizer.tokenize(title)
        tokens_b = None
        if content is not None:
            tokens_b = self.tokenizer.tokenize(content)
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

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        return [input_ids], [input_mask], [segment_ids]

    def predict(self, title, content=None):
        '''
            classification
        '''
        arr, mask, type_ids = self._convert_single_sentence(title, content, self.config.max_seq_length)
        yout = self.sess.run((self.y_out, self.y_prb), feed_dict={
            self.input_ids:arr,
            self.input_mask:mask,
            self.input_type_ids:type_ids
            })
        probs = [(unit,i) for i,unit in enumerate(yout[1][0])]
        probs = sorted(probs, key=lambda s:s[0], reverse=True)
        #return self.label_map[yout[0][0]]
        return [self.label_map[probs[0][1]],
                self.label_map[probs[1][1]],
                self.label_map[probs[2][1]]
               ]
        #return [(self.label_map[probs[0][1]], float(probs[0][0])),
        #        (self.label_map[probs[1][1]], float(probs[1][0])),
        #        (self.label_map[probs[2][1]], float(probs[2][0]))
        #       ]
        #res = {
        #    self.label_map[probs[0][1]]: float(probs[0][0]),
        #    self.label_map[probs[1][1]]: float(probs[1][0]),
        #    self.label_map[probs[2][1]]: float(probs[2][0])
        #}
        return res

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


if __name__ == '__main__':
    s = '哈登狂砍36分，火箭还是输球 今天上午火箭季后赛对阵勇士'
    model = BertSimple()
    while True:
        s = input('输入title + content\n')
        if s == 'q':
            break
        lists = s.split('@@@')
        y = model.predict(lists[0], lists[1])
        print json.dumps(y)
