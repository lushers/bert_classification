#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2019 lushers, Inc. All Rights Reserved
#
########################################################################
"""
File: evaluation.py
Author: lushers 
Date: 2019/07/05 16:15:48
"""
import time
import numpy as np
import json
import codecs
from sklearn.metrics import accuracy_score
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix, classification_report
from bert_simple import BertSimple

class Evaluation(object):
    """
        测试simple-bert 预测的结果
    """
    def __init__(self, label_name):
        self.label_name = label_name
        self.labels = []
        self.label_map = {}
        self.label_map_rev = {}
        self.y_true = []
        self.y_predict = []
        self.x = []

    def get_label(self):
        with codecs.open(self.label_name, 'r', 'utf-8') as fd:
            for line in fd:
                self.labels.append(line.strip())
            for (i, label) in enumerate(self.labels):
                self.label_map[label] = i
                self.label_map_rev[i] = label

    def read_test_file(self, file_name):
        with codecs.open(file_name, 'r', 'utf-8') as fd:
            for line in fd:
                lists = line.strip().split('\t|@@@|\t')
                if len(lists) != 3:
                    continue
                if lists[0] == u'影视':
                    lists[0] = u'娱乐'
                self.y_true.append(self.label_map[lists[0]])
                lists[1] = lists[1].replace(u'\n', u'').lstrip(u'。')
                lists[2] = lists[2].replace(u'\n', u'').lstrip(u'。')
                if len(lists[2]) < 256:
                    lists[2] = self._append_doc(lists[2], 256)
                self.x.append(lists[1] + '@@@' + lists[2])

    def _append_doc(self, lists, length):
        new_list = []
        lens = len(lists)
        for i in range(length):
            new_list.append(lists[i % lens])
        return u"".join(new_list)

    def read_json_file(self, file_name):
        try:
            with open(file_name, 'r') as fd:
                test_data = json.load(fd)
                for unit in test_data:
                    if unit['classes'] == u'影视':
                        unit['classes'] = u'娱乐'
                    self.y_true.append(self.label_map[unit['classes']])
                    self.x.append(unit['title'] + '@@@' + unit['content'])
        except Exception as ex:
            print ex

    def predict(self, cls, file_name):
        for line in self.x:
            lists = line.split('@@@')
            ress = cls.predict(lists[0], lists[1])
            self.y_predict.append([
                self.label_map[ress[0]],
                self.label_map[ress[1]],
                self.label_map[ress[2]]]
                #self.label_map[cls.predict(lists[0], lists[1]).decode('utf-8')]
            )
            #y_predict.append(cls.predict(line))
        with open(file_name, 'w') as fd:
            for line in self.y_predict:
                fd.write(str(line[0]) + ','+ str(line[1]) + ','+ str(line[2])+ '\n')

if __name__ == '__main__':
    eva = Evaluation('labels.txt')
    eva.get_label()
    eva.read_test_file('0818_26.text.now')
    #eva.read_json_file('/data8/nlp/trainset/weibo_hot/weibo_content_model_v5/validationSet/long_text_test_all.json')
    #with open('textAB_top1_200_res_.txt') as fd:
    #    for line in fd:
    #        lists = [int(unit) for unit in line.strip().split(',')]
    #        eva.y_predict.append(lists)
    cls = BertSimple()
    begin = time.time()
    eva.predict(cls, 'textAB_top1_oiio_sina_fine02.txt')
    pre = time.time()
    print (str(pre-begin).encode('utf-8'))
    y_true = eva.y_true
    y_predict = eva.y_predict
    labels = eva.labels
    flag = 'top1'
    for i, u in enumerate(eva.x):
        if int(y_true[i]) != y_predict[i][0]:
            strs = ''.join(u.strip().split('\n'))
            outs = eva.label_map_rev[y_predict[i][0]] + '\t' \
                + eva.label_map_rev[y_predict[i][1]] + '\t'  \
                + eva.label_map_rev[y_predict[i][2]] + '\t'  \
                + eva.label_map_rev[int(y_true[i])] + '\t' + strs
            outs = outs.encode('utf-8')
            print outs
        if flag == 'top3':
            if y_true[i] in y_predict[i]:
                y_predict[i] = y_true[i]
            else:
                y_predict[i] = y_predict[i][0]
        else:
            #if len(u) < 100 and y_predict[i][0] == 26:
            #    y_predict[i][0] = y_predict[i][1]
            y_predict[i] = y_predict[i][0]
    acc = accuracy_score(y_true, y_predict)
    macro_p = metrics.precision_score(y_true, y_predict, average='macro')
    micro_p = metrics.precision_score(y_true, y_predict, average='micro')
    con_max = confusion_matrix(y_true, y_predict)
    cls_rep = classification_report(y_true, y_predict, target_names=labels)
    #print acc, macro_p, micro_p
    print 'con_max' + '\t' + '\t'.join(labels).encode('utf-8')
    #print con_max
    for i, unit in enumerate(con_max):
        print labels[i].encode('utf-8') + '\t' + '\t'.join([str(n).encode('utf-8') for n in unit])
    print cls_rep.encode('utf-8')
