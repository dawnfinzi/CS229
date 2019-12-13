"""
Optimization utilities

Eshed Margalit & Dawn Finzi
"""

import ipdb
import tensorflow as tf
import numpy as np
from pprint import pprint

import transformations as xforms
from utils import norm_image
from losses import total_variation_loss


def preprocess(
    images,
):
    """
    Takes in an image tensor and completes the Lucid preprocessing steps
    Inputs
        Tensorflow variable of shape (1, image_resolution, image_resolution, 3)
    Outputs
        Tensorflow variable of shape (1, image_resolution, image_resolution, 3)
    """
    scales = [1 + (i - 5) / 50. for i in range(11)]
    angles = list(range(-10, 11)) + 5 * [0]

    images = xforms.pad(images, pad_amount=12)
    images = xforms.jitter(images, jitter_amount=8)
    images = xforms.random_scale(images, scales)
    images = xforms.random_rotate(images, angles)
    images = xforms.jitter(images, jitter_amount=4)

    return images

def get_network_aspect(
    params,
    layer,
):
    """
    Returns the target aspect of the network to optimize
    Inputs
        params (dict): keys include
            - "channel": which channel to do optimization for
            - "unit_index" ([row, col]): if does not exist, optimizes for whole channel. If does exist,
                optimizes only for that unit.
        layer (tensor): layer of network 
    Outputs
        target (tensor): network aspect to optimize for
    """
    if 'unit_index' not in params:
        if len(layer.get_shape().as_list()) == 4:
            target = layer[0, :, :, params['channel']]
        else:
            target = layer[0, params['channel']]
    else:
        unit_row, unit_col = params['unit_index']
        if len(layer.get_shape().as_list()) == 4:
            target = layer[0, unit_row, unit_col, params['channel']]
        else:
            print("Warning: both unit index and channel were provided for an FC layer, which conflicts (channels _are_ units). Going with channel instead of index") 
            target = layer[0, params['channel']]
    
    return target

def get_optimal_image(
    model_fn,
    model_kwargs,
    checkpoint_path,
    params,
    preproc=True,
    layer_name=None,
    image_resolution=128
):
    """
    Does gradient ascent to get the optimal image for a given model
    Inputs
        model_fn (fn): function to which image tensor (batch x h x w x c) can be passed
        model_kwargs (dict): other keyword arguments for the model function
        checkpoint_path (str): where to find the model checkpoint
        params (dict): keys include
            - "channel": which channel to do optimization for
            - "learning rate"
            - "regularization"
            - "steps": how many steps to run for
            - optional: "unit_index"
            - optional: "loss" (str) - what loss function to use (default is just L2 regulariation). 
            - optional: "loss_lambda" (int) - constant to scale additional loss by (default is 1)
        preproc (bool): whether or not to preprocess the images ala lucid (default is True)
        layer_name (str): which to layer to get image for
        image_resolution (int): how many pixels to make the image on each side
        
    Outputs
        optimal image (224 x 224 x 3)
    """
    # set up model
    tf.reset_default_graph()
    init = tf.random_uniform_initializer(minval=0, maxval=1) #initialize random noise
    reg = tf.contrib.layers.l2_regularizer(scale=params['regularization']) #setup L2 reg

    # set up the image variable
    image_shape = (1, image_resolution, image_resolution, 3)
    images = tf.get_variable("images", image_shape, initializer=init, regularizer=reg)

    # preprocess the images
    if preproc is True:
        images = preprocess(images)
        
    # get features for a given layer from a given model
    tensor_name = params.get('tensor_name', None)
    layer = model_fn(images, layer_name=layer_name, tensor_name=tensor_name, **model_kwargs)

    # extract specified aspect of the netowork to optimize from conv or fc layer
    target = get_network_aspect(params, layer)

    # set up loss function
    if 'loss' not in params: #loss is None:
        total_reg = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) #basic L2 loss
    else:
        tv_loss = total_variation_loss(images)
        if 'loss_lambda' in params:
            scale_loss = tf.constant(params['loss_lambda'])
        else:
            scale_loss = tf.constant(1.0)
        # check and fix data type
        scale_loss = tf.cast(scale_loss, tv_loss.dtype)
        total_reg = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) + tf.scalar_mul(scale_loss,tv_loss)
    loss_tensor = tf.negative(tf.reduce_mean(target)) + total_reg

    # set up optimizer
    lr_tensor = tf.constant(params['learning_rate'])

    # restrict trainable variables to the image itself
    train_vars = [
        var for var in tf.trainable_variables() if 'images' in var.name
    ]
    train_op = tf.train.AdamOptimizer(lr_tensor).minimize(loss_tensor, var_list=train_vars)

    ## Start the session, initialize variables and restore the model weights
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    all_variables = tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES)
    # exclude the new image variables
    temp_saver = tf.train.Saver(
        var_list=[v for v in all_variables if "images" not in v.name and "beta" not in v.name]
    )
    temp_saver.restore(sess, checkpoint_path) #restore model weigths

    ## Main Loop
    loss_list = list()
    for i in range(params['steps']):
        loss_list.append(sess.run(loss_tensor)) #keep track of the loss over steps
        sess.run(train_op) #optimize image

    final_image = sess.run(images)
    return norm_image(final_image.squeeze()), loss_list
