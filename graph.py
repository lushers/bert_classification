#!/usr/bin/env python
#coding:utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import tempfile
import os
import json

import tensorflow as tf
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference
from bert import modeling
import contextlib
from simple_bert import SimpleBert

def convert_variables_to_constants(sess,
                                   input_graph_def,
                                   output_node_names,
                                   variable_names_whitelist=None,
                                   variable_names_blacklist=None,
                                   use_fp16=False):
    from tensorflow.python.framework.graph_util_impl import extract_sub_graph
    from tensorflow.core.framework import graph_pb2
    from tensorflow.core.framework import node_def_pb2
    from tensorflow.core.framework import attr_value_pb2
    from tensorflow.core.framework import types_pb2
    from tensorflow.python.framework import tensor_util

    def patch_dtype(input_node, field_name, output_node):
        if use_fp16 and (field_name in input_node.attr) and (input_node.attr[field_name].type == types_pb2.DT_FLOAT):
            output_node.attr[field_name].CopyFrom(attr_value_pb2.AttrValue(type=types_pb2.DT_HALF))

    inference_graph = extract_sub_graph(input_graph_def, output_node_names)

    variable_names = []
    variable_dict_names = []
    for node in inference_graph.node:
        if node.op in ["Variable", "VariableV2", "VarHandleOp"]:
            variable_name = node.name
            if ((variable_names_whitelist is not None and
                 variable_name not in variable_names_whitelist) or
                    (variable_names_blacklist is not None and
                     variable_name in variable_names_blacklist)):
                continue
            variable_dict_names.append(variable_name)
            if node.op == "VarHandleOp":
                variable_names.append(variable_name + "/Read/ReadVariableOp:0")
            else:
                variable_names.append(variable_name + ":0")
    if variable_names:
        returned_variables = sess.run(variable_names)
    else:
        returned_variables = []
    found_variables = dict(zip(variable_dict_names, returned_variables))

    output_graph_def = graph_pb2.GraphDef()
    how_many_converted = 0
    for input_node in inference_graph.node:
        output_node = node_def_pb2.NodeDef()
        if input_node.name in found_variables:
            output_node.op = "Const"
            output_node.name = input_node.name
            dtype = input_node.attr["dtype"]
            data = found_variables[input_node.name]

            if use_fp16 and dtype.type == types_pb2.DT_FLOAT:
                output_node.attr["value"].CopyFrom(
                    attr_value_pb2.AttrValue(
                        tensor=tensor_util.make_tensor_proto(data.astype('float16'),
                                                             dtype=types_pb2.DT_HALF,
                                                             shape=data.shape)))
            else:
                output_node.attr["dtype"].CopyFrom(dtype)
                output_node.attr["value"].CopyFrom(attr_value_pb2.AttrValue(
                    tensor=tensor_util.make_tensor_proto(data, dtype=dtype.type,
                                                         shape=data.shape)))
            how_many_converted += 1
        elif input_node.op == "ReadVariableOp" and (input_node.input[0] in found_variables):
            # placeholder nodes
            # print('- %s | %s ' % (input_node.name, input_node.attr["dtype"]))
            output_node.op = "Identity"
            output_node.name = input_node.name
            output_node.input.extend([input_node.input[0]])
            output_node.attr["T"].CopyFrom(input_node.attr["dtype"])
            if "_class" in input_node.attr:
                output_node.attr["_class"].CopyFrom(input_node.attr["_class"])
        else:
            # mostly op nodes
            output_node.CopyFrom(input_node)

        patch_dtype(input_node, 'dtype', output_node)
        patch_dtype(input_node, 'T', output_node)
        patch_dtype(input_node, 'DstT', output_node)
        patch_dtype(input_node, 'SrcT', output_node)
        patch_dtype(input_node, 'Tparams', output_node)

        if use_fp16 and ('value' in output_node.attr) and (
                output_node.attr['value'].tensor.dtype == types_pb2.DT_FLOAT):
            # hard-coded value need to be converted as well
            output_node.attr['value'].CopyFrom(attr_value_pb2.AttrValue(
                tensor=tensor_util.make_tensor_proto(
                    output_node.attr['value'].tensor.float_val[0],
                    dtype=types_pb2.DT_HALF)))

        output_graph_def.node.extend([output_node])

    output_graph_def.library.CopyFrom(inference_graph.library)
    return output_graph_def


config = tf.ConfigProto(device_count={'GPU': 0}, allow_soft_placement=True)

init_checkpoint = \
'./model_bak/model.ckpt-300000'
#'./model_sina_ab/model.ckpt-37342'
#'/data8/qinchuan1/workspace/simple_bert/temp/model.ckpt-19000'
#'/data8/qinchuan1/workspace/simple_bert/model_output_ab/model.ckpt-218900'
#'/data8/qinchuan1/workspace/simple_bert/model_256_bak/model.ckpt-153500'
#'/data2/sina_recmd/qinchuan1/workspace/simple_bert/model_output_512/model.ckpt-80800'

input_ids = tf.placeholder(shape=[None, None], dtype=tf.int32, name='input_ids')
input_mask = tf.placeholder(shape=[None, None], dtype=tf.int32, name='input_mask')
segment_ids = tf.placeholder(shape= [None, None], dtype=tf.int32, name='input_type_ids')
labels = tf.placeholder(shape=[None], dtype=tf.int64, name='labels')

model = SimpleBert()

loss, per, logits, probabilities, predict_res = model.create_model(
        is_training=False,
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        labels=labels,
        num_labels=model.num_labels,
        use_one_hot_embeddings=False)

tvars = tf.trainable_variables()
(assignment_map, initialized_variable_names
) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)

tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

minus_mask = lambda x, m: x - tf.expand_dims(1.0 - m, axis=-1) * 1e30
mul_mask = lambda x, m: x * tf.expand_dims(m, axis=-1)
masked_reduce_max = lambda x, m: tf.reduce_max(minus_mask(x, m), axis=1)
masked_reduce_mean = lambda x, m: tf.reduce_sum(mul_mask(x, m), axis=1) / (
                            tf.reduce_sum(m, axis=1, keepdims=True) + 1e-10)

with tf.variable_scope("pooling"):
    predict = tf.identity(predict_res, 'predict')
    probabilities = tf.identity(probabilities, 'probabilities')
    output_tensors = [predict, probabilities]
    tmp_g = tf.get_default_graph().as_graph_def()

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    tmp_g = convert_variables_to_constants(sess, tmp_g, [n.name[:-2] for n in output_tensors])

    tmp_file = tempfile.NamedTemporaryFile('w', delete=False).name
    with tf.gfile.GFile('model_ab_sina_512.pb', 'wb') as f:
        f.write(tmp_g.SerializeToString())
