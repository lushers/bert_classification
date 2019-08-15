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
        # for model
        self.json_config = "chinese_L-12_H-768_A-12/bert_config.json"
        self.vocab_file = "chinese_L-12_H-768_A-12/vocab.txt"
        self.init_checkpoint = "chinese_L-12_H-768_A-12/bert_model.ckpt"
        self.batch_size = 16
        self.num_labels = 37 # classes
        self.is_training = True
        self.max_seq_length = 512
        self.learning_rate = 5e-5
        self.train_examples = 4800000
        self.epochs = 2
        # for data
        self.output_dir = "./data/sina"
        self.data_dir = "./data/sina"
        self.task_name = "mine_ab"

