#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2019 lushers, Inc. All Rights Reserved
#
########################################################################
"""
File: SimpleBert.py
Author: lushers
Date: 2019/06/20 22:24:21
"""
import sys
import os
import tensorflow as tf
import numpy as np
from bert import modeling, tokenization, optimization
from bert_utils import *
from config import Config

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

class SimpleBert(object):
    """
        BertModel for classification finetuing
    """
    def __init__(self):
        self.config = Config()
        self.batch_size = self.config.batch_size
        self.max_seq_length = self.config.max_seq_length
        self.num_labels = self.config.num_labels
        self.learning_rate = self.config.learning_rate
        self.train_examples = self.config.train_examples
        self.epochs = self.config.epochs

        self.init_checkpoint = self.config.init_checkpoint
        self.vocab_file = self.config.vocab_file
        self.json_config = self.config.json_config
        self.output_dir = self.config.output_dir
        self.data_dir = self.config.data_dir
        self.task_name = self.config.task_name
        self.bert_config = modeling.BertConfig.from_json_file(self.json_config)

        self.warmup_proportion = 0.1
        self.num_train_steps = int(self.train_examples/self.batch_size*self.epochs)
        self.num_warmup_steps = int(self.num_train_steps * self.warmup_proportion)
        self.run_config = tf.estimator.RunConfig(
                model_dir='model_sina_ab_maskfc',
                save_summary_steps=100,
                keep_checkpoint_max=5,
                save_checkpoints_steps=100)

    def create_model(self, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
        """
            finetuing bert
        """
        # 创建bert模型
        model = modeling.BertModel(
            config=self.bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=False  # without TPU
        )
        # simple use [cls]
        #output_layer = model.get_all_encoder_layers()[3]
        #first_token_tensor = tf.squeeze(output_layer[:, 0:1, :], axis=1)
        output_layer = model.get_pooled_output()
        hidden_size = output_layer.shape[-1].value
        # add dense (try ? )
        #dense_layer = tf.layers.dense(first_token_tensor, 1024, activation='relu')
        #dense_layer = tf.identity(dense_layer, name='dense_layer')

        #hidden_size = dense_layer.shape[-1].value
        #output_layer = dense_layer

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
        """
            model_fn for Estimator
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
            False)

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
            train_op = optimization.create_optimizer(
                loss,
                params['learning_rate'],
                params['num_train_steps'],
                params['num_warmup_steps'],
                False)
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
        eval_steps = 1000  # 1000 * batch_size
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
                'batch_size':self.batch_size
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
        seq_length = self.max_seq_length

        feature_spec = {
            "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
            "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "label_ids": tf.FixedLenFeature([], tf.int64)
        }
        def serving_input_receiver_fn():
            """
            This is used to define inputs to serve the model.
            :return: ServingInputReciever
            """
            serialized_tf_example = tf.placeholder(dtype=tf.string, shape=[self.batch_size], name='input_example_tensor')
            receiver_tensors = {'examples': serialized_tf_example}
            features = tf.parse_example(serialized_tf_example, feature_spec)
            return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)
        # Define evaluating spec.
        latest_exporter = tf.estimator.LatestExporter(
            name="models",
            serving_input_receiver_fn=serving_input_receiver_fn,
            exports_to_keep=5)
        best_exporter = tf.estimator.BestExporter(
            serving_input_receiver_fn=serving_input_receiver_fn,
            exports_to_keep=1)
        exporters = [latest_exporter, best_exporter]

        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=self.num_train_steps)
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=None)
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
        estimator.export_savedmodel('./saved_models', serving_input_receiver_fn, strip_default_attrs=True)

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
        """
            gen data and process
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
    #simple_bert.export_saved_model()
    #simple_bert.train(train_file)
    simple_bert.train_and_eval(train_file, eval_file)
    #simple_bert.predict(predict_file)
    #simple_bert.eval(eval_file)
    #simple_bert.train(train_file)
    #simple_bert.gen_data('sina/train_sina.record', 'sina/eval_sina.record')
