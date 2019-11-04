#!/usr/bin/env python
#coding:utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import os

from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
import tensorflow as tf
import contextlib

os.environ['CUDA_VISIBLE_DEVICES'] = '2'


export_dir = 'serving_test/288000'
graph_pb = './model_ab_sine_288000.pb'
builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
with tf.gfile.GFile(graph_pb, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Session(graph=tf.Graph()) as sess:
    tf.import_graph_def(graph_def, name="")
    g = tf.get_default_graph()
    input_ids = g.get_tensor_by_name('input_ids:0')
    input_mask = g.get_tensor_by_name('input_mask:0')
    segment_ids = g.get_tensor_by_name('input_type_ids:0')
    probabilities = g.get_tensor_by_name('pooling/probabilities:0')
    predict = g.get_tensor_by_name('pooling/predict:0')

    tensor_info_inputs = {
        'input_ids': tf.saved_model.utils.build_tensor_info(input_ids),
        'input_mask': tf.saved_model.utils.build_tensor_info(input_mask),
        'segment_ids': tf.saved_model.utils.build_tensor_info(segment_ids)
    }
    tensor_info_outputs = {
        "predict": tf.saved_model.utils.build_tensor_info(predict),
        "probabilities": tf.saved_model.utils.build_tensor_info(probabilities)
    }
    classify_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs=tensor_info_inputs,
            outputs=tensor_info_outputs,
            method_name="tensorflow/serving/predict")
    )

    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            classify_signature
        })
builder.save()
