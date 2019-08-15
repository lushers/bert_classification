#!/usr/bin/env python
#coding:utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

class Config():
    def __init__(self):
        self.model_path = './model_ab_sina_512.pb'
        #句子最大长度
        self.max_seq_length = 512
        self.vocab_file = './chinese_L-12_H-768_A-12/vocab.txt'
