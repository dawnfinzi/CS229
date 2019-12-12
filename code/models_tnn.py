"""
Defines CNN architectures
(Credit to Eshed Margalit)
"""
import ipdb
import os
import sys
sys.path.append('../models/')
from tnn import main as tnn_main
import tensorflow as tf

def tnn_model_func(inputs, **kwargs):
    if isinstance(inputs, dict):
        images = inputs["images"]
    else:
        images = inputs

    base_name = kwargs['json_fpath']
    ims = tf.identity(tf.cast(images, dtype=tf.float32), name='split')
    # batch_size = ims.get_shape().as_list()[0]
    batch_size = kwargs["batch_size"]
    
    with tf.variable_scope('tnn_model'):
        if '.json' not in base_name:
            base_name += '.json'

        G = tnn_main.graph_from_json(base_name)

        # initialize graph structure
        tnn_main.init_nodes(
            G,
            input_nodes=['conv1'],
            batch_size=batch_size,
            channel_op='concat'
        )
        
        # unroll graph
        tnn_main.unroll_tf(G, input_seq={'conv1': ims}, ntimes=1)

        outputs = {}
        for l in ['conv' + str(t) for t in range(1,11)]:
            outputs[l] = G.node[l]['outputs'][-1]

        outputs['logits'] = G.node['imnetds']['outputs'][-1]

    return outputs

def tnn_wrapper(images, layer_name, **kwargs):
    model_outputs = tnn_model_func(images, **kwargs)
    layer = model_outputs[layer_name]
    return layer

def tnn_no_fc_wrapper(images, layer_name, **kwargs):
    model_outputs = tnn_model_func(images, **kwargs)
    model_outputs.pop('logits')
    layer = model_outputs[layer_name]
    return layer
