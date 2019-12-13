"""
Defines Alexnet architectures
"""

import ipdb
import tensorflow as tf
from tfutils import model_tool


def alexnet(images, train=True, norm=True, seed=0, **kwargs):
    """
    Alexnet
    """
    m = model_tool.ConvNet(seed=seed)

    conv_kwargs = {"add_bn": False, "init": "xavier", "weight_decay": 0.0001}
    pool_kwargs = {"pool_type": "maxpool"}
    fc_kwargs = {"init": "trunc_norm", "weight_decay": 0.0001, "stddev": 0.01}

    dropout = 0.5 if train else None

    m.conv(96, 11, 4, padding="VALID", layer="conv1",
           in_layer=images, **conv_kwargs)
    if norm:
        m.lrn(depth_radius=5, bias=1, alpha=0.0001, beta=0.75, layer="lrn1")
    m.pool(3, 2, layer="pool1", **pool_kwargs)

    m.conv(256, 5, 1, layer="conv2", **conv_kwargs)
    if norm:
        m.lrn(depth_radius=5, bias=1, alpha=0.0001, beta=0.75, layer="lrn2")
    m.pool(3, 2, layer="pool2", **pool_kwargs)

    m.conv(384, 3, 1, layer="conv3", **conv_kwargs)
    m.conv(384, 3, 1, layer="conv4", **conv_kwargs)

    m.conv(256, 3, 1, layer="conv5", **conv_kwargs)
    m.pool(3, 2, layer="pool5", **pool_kwargs)

    m.fc(4096, dropout=dropout, bias=0.1, layer="fc6", **fc_kwargs)
    m.fc(4096, dropout=dropout, bias=0.1, layer="fc7", **fc_kwargs)
    m.fc(1000, activation=None, dropout=None, bias=0, layer="fc8", **fc_kwargs)

    return m

def alexnet_no_fc(images, train=True, norm=True, seed=0, **kwargs):
    """
    Alexnet
    """
    m = model_tool.ConvNet(seed=seed)

    conv_kwargs = {"add_bn": False, "init": "xavier", "weight_decay": 0.0001}
    pool_kwargs = {"pool_type": "maxpool"}
    fc_kwargs = {"init": "trunc_norm", "weight_decay": 0.0001, "stddev": 0.01}

    dropout = 0.5 if train else None

    m.conv(96, 11, 4, padding="VALID", layer="conv1",
           in_layer=images, **conv_kwargs)
    if norm:
        m.lrn(depth_radius=5, bias=1, alpha=0.0001, beta=0.75, layer="lrn1")
    m.pool(3, 2, layer="pool1", **pool_kwargs)

    m.conv(256, 5, 1, layer="conv2", **conv_kwargs)
    if norm:
        m.lrn(depth_radius=5, bias=1, alpha=0.0001, beta=0.75, layer="lrn2")
    m.pool(3, 2, layer="pool2", **pool_kwargs)

    m.conv(384, 3, 1, layer="conv3", **conv_kwargs)
    m.conv(384, 3, 1, layer="conv4", **conv_kwargs)

    m.conv(256, 3, 1, layer="conv5", **conv_kwargs)
    m.pool(3, 2, layer="pool5", **pool_kwargs)

    return m

def alexnet_wrapper(images, layer_name=None, tensor_name=None, **kwargs):
    assert tensor_name is not None or layer_name is not None, "must provide either a tensor name or a layer name"
    model = alexnet(images, **kwargs)
    if tensor_name is not None:
        output = tf.get_default_graph().get_tensor_by_name(tensor_name + ":0")
    else:
        output = model.layers[layer_name]
    return output

def alexnet_no_fc_wrapper(images, layer_name=None, tensor_name=None, **kwargs):
    assert tensor_name is not None or layer_name is not None, "must provide either a tensor name or a layer name"
    model = alexnet_no_fc(images, **kwargs)
    if tensor_name is not None:
        output = tf.get_default_graph().get_tensor_by_name(tensor_name + ":0")
    else:
        output = model.layers[layer_name]
    return output