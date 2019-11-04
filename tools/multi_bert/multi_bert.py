#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2019 lushers Inc. All Rights Reserved
#
########################################################################
"""
File: SimpleBert.py
Author: lushers 
Date: 2019/06/20 22:24:21
"""
import sys
import os

_cur_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(_cur_dir + '/bert_multi_gpu')
import tensorflow as tf
import numpy as np
from bert_multi_gpu import modeling, tokenization, optimization, custom_optimization
from bert_utils import *
from config import Config

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,3'

class SimpleBert(object):
    """
        BertModel for classification finetuing
    """
    def __init__(self):
        # basic config
        self.config = Config()
        self.max_seq_length = self.config.max_seq_length
        self.num_labels = self.config.num_labels
        self.fp16 = self.config.fp16
        self.train_examples = self.config.train_examples
        # data/model config
        self.init_checkpoint = self.config.init_checkpoint
        self.vocab_file = self.config.vocab_file
        self.json_config = self.config.json_config
        self.output_dir = self.config.output_dir
        self.data_dir = self.config.data_dir
        self.task_name = self.config.task_name
        self.bert_config = modeling.BertConfig.from_json_file(self.json_config)
        # train config
        self.use_gpu = self.config.use_gpu
        self.num_gpu_cores = self.config.num_gpu_cores
        self.learning_rate = self.config.learning_rate
        self.batch_size = self.config.batch_size
        self.epochs = self.config.epochs
        self.warmup_proportion = 0.1
        self.num_train_steps = int(self.train_examples / (self.batch_size*self.num_gpu_cores) * self.epochs)
        self.num_warmup_steps = int(self.num_train_steps * self.warmup_proportion)
        self.run_config = self.init_run_config(self.use_gpu, self.num_gpu_cores)

    def init_run_config(self, use_gpu, num_gpu_cores):
        """init run_config for multi_gpu
        """
        log_every_n_steps = 50
        if use_gpu and num_gpu_cores >= 2:
            tf.logging.info("use multi gpu config")
            dist_strategy = tf.contrib.distribute.MirroredStrategy(
                num_gpus=num_gpu_cores,
                #cross_device_ops=AllReduceCrossDeviceOps('nccl', num_packs=FLAGS.num_gpu_cores),
                #cross_tower_ops=tf.contrib.distribute.AllReduceCrossTowerOps('nccl', num_packs=FLAGS.num_gpu_cores)
                cross_tower_ops=tf.contrib.distribute.AllReduceCrossTowerOps('hierarchical_copy', num_packs=num_gpu_cores),
            )
            run_config = tf.estimator.RunConfig(
                train_distribute=dist_strategy,
                eval_distribute=dist_strategy,
                log_step_count_steps=log_every_n_steps,
                model_dir='model_512_roberta',
                save_checkpoints_steps=100,
                keep_checkpoint_max=5,
                save_summary_steps=100
            )
        else:
            tf.logging.info("use normal config (cpu/one-gpu)")
            run_config = tf.estimator.RunConfig(
                log_step_count_steps=log_every_n_steps,
                model_dir='model_512_roberta',
                save_checkpoints_steps=100,
                keep_checkpoint_max=5,
                save_summary_steps=100
            )
        return run_config

    def create_model(self, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings, fp16):
        """creat bert model
            fine-tuning:
                (1) 经过测试,对于分类任务,增加全连接层的大小不能提高分类的精确率
                (2) 减少transfomer的层数, 对于参数的减少较为明显,
                对于效果（37）分类在训练、验证集上变化不大(98%-96% 2-epoch),
                但是测试集上的badcase明显增多
        """
        comp_type = tf.float16 if fp16 else tf.float32
        # 创建bert模型
        model = modeling.BertModel(
            config=self.bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=False,  # without TPU
            comp_type=comp_type
        )
        output_layer = model.get_pooled_output()
        hidden_size = output_layer.shape[-1].value

        # add softmax layer
        output_weights = tf.get_variable(
            "output_weights", [num_labels, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable(
            "output_bias", [num_labels], initializer=tf.zeros_initializer())

        with tf.variable_scope("loss"):
            if is_training:
                output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            probabilities = tf.nn.softmax(logits, name="probabilities")
            log_probs = tf.nn.log_softmax(logits, axis=-1)

            one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            loss = tf.reduce_mean(per_example_loss)
            predict = tf.argmax(probabilities, axis=1, name="predict")
            # acc = tf.reduce_mean(tf.cast(tf.equal(labels, tf.cast(predict, dtype=tf.int32)), "float"), name="accuracy")

        return (loss, per_example_loss, logits, probabilities, predict)

    def ml_model_fn(self, features, labels, mode, params):
        """model_fn for Estimator
        """
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        scaffold_fn = None

        #load basic model
        # total_loss ... loss
        (loss, per_example_loss, logits, probabilities, predict) = self.create_model(
            is_training,
            input_ids,
            input_mask,
            segment_ids,
            label_ids,
            self.num_labels,
            False,
            self.fp16
        )

        tvars = tf.trainable_variables()
        (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, self.init_checkpoint)
        # no tpu
        tf.train.init_from_checkpoint(self.init_checkpoint, assignment_map)
        # summary
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        accuracy = tf.metrics.accuracy(labels=label_ids, predictions=predictions)
        tf.summary.scalar('accuracy', accuracy[1])
        export_outputs = {
            'predict_output': tf.estimator.export.PredictOutput({
                'predict': predict,
                'probabilities': probabilities
            })
        }
        if mode == tf.estimator.ModeKeys.TRAIN:
            # 单机多卡
            if self.use_gpu and int(self.num_gpu_cores) >= 2:
                train_op = custom_optimization.create_optimizer(
                    loss,
                    params['learning_rate'],
                    params['num_train_steps'],
                    params['num_warmup_steps'],
                    params['fp16']
                )
            else:
                train_op = optimization.create_optimizer(
                    loss,
                    params['learning_rate'],
                    params['num_train_steps'],
                    params['num_warmup_steps'],
                    False,
                    params['fp16']
                )
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
                export_outputs=export_outputs,
                scaffold=scaffold_fn)
        if mode == tf.estimator.ModeKeys.EVAL:
            eval_loss = tf.metrics.mean(values=per_example_loss)
            eval_metrics = {
                "eval_accuracy":accuracy,
                "eval_loss":eval_loss
            }
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                eval_metric_ops=eval_metrics,
                export_outputs=export_outputs,
                scaffold=scaffold_fn)
        if mode == tf.estimator.ModeKeys.PREDICT:
            predicted_classes = tf.argmax(logits, 1)
            predictions = {
                'class_ids': predicted_classes[:, tf.newaxis],
                'probabilities': probabilities,
                'logits': logits
            }
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                export_outputs=export_outputs,
                scaffold=scaffold_fn)
        return output_spec

    def train(self, input_file):
        model_fn = self.ml_model_fn
        estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            config=self.run_config,
            params={
                'num_labels':self.num_labels,
                'learning_rate':self.learning_rate,
                'num_train_steps':self.num_train_steps,
                'num_warmup_steps':self.num_warmup_steps,
                'batch_size':self.batch_size
            })
        train_input_fn = file_based_input_fn_builder(
                input_file=input_file,
                seq_length=self.max_seq_length,
                is_training=True,
                drop_remainder=True)
        def serving_input_receiver_fn():
            feature_spec = {
                'input_ids' : tf.placeholder(shape=[None, self.max_seq_length], dtype=tf.int32, name='input_ids'),
                'input_mask' : tf.placeholder(shape=[None, self.max_seq_length], dtype=tf.int32, name='input_mask'),
                'segment_ids' : tf.placeholder(shape= [None, self.max_seq_length], dtype=tf.int32, name='input_type_ids'),
                'label_ids' : tf.placeholder(shape=[None], dtype=tf.int64, name='labels')
            }
            return tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)()
        estimator.train(input_fn=train_input_fn, max_steps=100)
        estimator.export_savedmodel('./saved_models', serving_input_receiver_fn, strip_default_attrs=True)

    def eval(self, eval_file):
        model_fn = self.ml_model_fn
        estimator = tf.estimator.Estimator(
                model_fn=model_fn,
                config=self.run_config,
                params={'batch_size':self.batch_size})
        eval_steps = 300  # 1000 * batch_size
        eval_drop_remainder = False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=self.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)
        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
        tf.logging.info("***** result done !*****")
        output_eval_file = os.path.join(self.output_dir, "eval_results.v1.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    def train_and_eval(self, input_file, eval_file):
        model_fn = self.ml_model_fn
        estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            config=self.run_config,
            params={
                'num_labels':self.num_labels,
                'learning_rate':self.learning_rate,
                'num_train_steps':self.num_train_steps,
                'num_warmup_steps':self.num_warmup_steps,
                'batch_size':self.batch_size,
                'fp16':self.fp16
            })
        train_input_fn = file_based_input_fn_builder(
            input_file=input_file,
            seq_length=self.max_seq_length,
            is_training=True,
            drop_remainder=True)

        eval_drop_remainder = False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=self.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)
        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=self.num_train_steps)
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=None)
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    def predict(self, predict_file):
        model_fn = self.ml_model_fn
        estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            config=self.run_config,
            params={'batch_size':self.batch_size})
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=self.max_seq_length,
            is_training=False,
            drop_remainder=False)

        result = estimator.predict(input_fn=predict_input_fn)
        output_predict_file = os.path.join(self.output_dir, "test_results.tsv")
        with tf.gfile.GFile(output_predict_file, "w") as writer:
            num_written_lines = 0
            tf.logging.info("***** Predict results *****")
            for (i, prediction) in enumerate(result):
                probabilities = prediction["probabilities"]
                output_line = "\t".join(
                    str(class_probability)
                    for class_probability in probabilities) + "\n"
                writer.write(output_line)
                num_written_lines += 1

    def export_saved_model(self):
        model_fn = self.ml_model_fn
        estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            config=self.run_config,
            params={'batch_size':self.batch_size})
        seq_length = self.max_seq_length
        def serving_input_receiver_fn():
            feature_spec = {
                'input_ids' : tf.placeholder(shape=[None, self.max_seq_length], dtype=tf.int32, name='input_ids'),
                'input_mask' : tf.placeholder(shape=[None, self.max_seq_length], dtype=tf.int32, name='input_mask'),
                'segment_ids' : tf.placeholder(shape= [None, self.max_seq_length], dtype=tf.int32, name='input_type_ids'),
                'label_ids' : tf.placeholder(shape=[None], dtype=tf.int64, name='labels')
            }
            return tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)()
        estimator.export_savedmodel('./saved_models', serving_input_receiver_fn, strip_default_attrs=True)

    def gen_data(self, train_record, eval_record, test_record=None):
        """gen data and process
        """
        tokenizer = tokenization.FullTokenizer(
            vocab_file=self.vocab_file, do_lower_case=True)
        processors = {
            "mine" : MineProcessor,
            "mine_ab" : MineABProcessor
        }
        #tf.gfile.MakeDirs(self.output_dir)
        processor = processors[self.task_name]()
        label_list = processor.get_labels()
        # train
        train_examples = processor.get_train_examples(self.data_dir)
        train_file = os.path.join(self.output_dir, train_record)
        file_based_convert_examples_to_features(
            train_examples, label_list, self.max_seq_length, tokenizer, train_file)
        # dev
        eval_examples = processor.get_dev_examples(self.data_dir)
        eval_file = os.path.join(self.output_dir, eval_record)
        file_based_convert_examples_to_features(
            eval_examples, label_list, self.max_seq_length, tokenizer, eval_file)
        # predict
        #predict_examples = processor.get_test_examples(self.data_dir)
        #predict_file = os.path.join(self.output_dir, "predict.tf_record")
        #file_based_convert_examples_to_features(
        #    predict_examples, label_list, self.max_seq_length, tokenizer, predict_file)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    simple_bert = SimpleBert()
    train_file = simple_bert.config.output_dir + '/train_512_ab.tf_record'
    eval_file = simple_bert.config.output_dir + '/eval_512_ab.tf_record'
    #predict_file = simple_bert.config.output_dir + '/predict.tf_record'
    simple_bert.train_and_eval(train_file, eval_file)
