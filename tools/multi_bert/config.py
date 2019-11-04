#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2019 lushers, Inc. All Rights Reserved
#
########################################################################
"""
File: config.py
Author: lushers
Date: 2019/06/20 22:57:12
"""
class Config(object):
    """bert init config"""
    def __init__(self):
        # for fine-tuning model
        self.json_config = "roberta_wwm/bert_config.json"
        self.vocab_file = "roberta_wwm/vocab.txt"
        self.init_checkpoint = "roberta_wwm/bert_model.ckpt"
        # bert ori
        #self.json_config = "chinese_L-12_H-768_A-12/bert_config.json"
        #self.vocab_file = "chinese_L-12_H-768_A-12/vocab.txt"
        #self.init_checkpoint = "chinese_L-12_H-768_A-12/bert_model.ckpt"
        # best steps (one-epoch)
        #self.init_checkpoint = 'init_point/model.ckpt-300000'
        # extra config
        self.use_gpu = True
        self.num_gpu_cores = 3
        self.fp16 = False
        # basic config
        self.batch_size = 16
        self.num_labels = 37 # classes
        self.is_training = True
        self.max_seq_length = 512
        self.learning_rate = 5e-5
        self.train_examples = 4743640
        self.epochs = 1
        # for data
        self.output_dir = "10_1_clean_to/records"
        self.data_dir = "10_1_clean_to/records"
        self.task_name = "mine_ab"

