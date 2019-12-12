"""
Optimization utilities

Initial credit to Eshed Margalit
Edited by Dawn Finzi 12/2019
"""

import ipdb
import tensorflow as tf
import numpy as np
from pprint import pprint

import transformations as xforms

def norm_image(x):
    return (x - np.min(x))/np.ptp(x)

def total_variation_loss(image_tensor):
    """
    Total variation loss (TV)
    Higher TV indicates bigger differences between adjacent pixels
    """

    X = tf.squeeze(image_tensor)
    xdiff = X[1:, :, :] - X[:-1, :, :]
    ydiff = X[:, 1:, :] - X[:, :-1, :]

    xdiff_l2 = tf.sqrt(tf.reduce_sum(tf.square(xdiff)))
    ydiff_l2 = tf.sqrt(tf.reduce_sum(tf.square(ydiff)))

    tv_loss = xdiff_l2 + ydiff_l2
    return tv_loss

def get_optimal_image(
    model_fn,
    model_kwargs,
    checkpoint_path,
    params,
    loss=None,
    preproc=True,
    layer_name=None,
    image_resolution=128,
    unit_index=None,
):
    """
    Does gradient ascent to get the optimal image for a given model
    Inputs
        model_fn (fn): function to which image tensor (batch x h x w x c) can be passed
        model_kwargs (dict): other keyword arguments for the model function
        checkpoint_path (str): where to find the model checkpoint
        layer_name (str): which to layer to get image for
        params (dict): keys include
            - "channel": which channel to do optimization for
            - "learning rate"
            - "regularization"
            - "steps": how many steps to run for
        loss (str): what loss function to use (default is just L2 regulariation). 
            Currently, only TV loss is implemented otherwise.
        preproc (bool): whether or not to preprocess the images ala lucid (default is True)
        image_resolution (int): how many pixels to make the image on each side
        unit_index ([row, col]): if None, optimizes for whole channel. If not None,
            optimizes only for that unit.
    Outputs
        optimal image (224 x 224 x 3)
    """
    # set up model
    tf.reset_default_graph()
    init = tf.random_uniform_initializer(minval=0, maxval=1)
    reg = tf.contrib.layers.l2_regularizer(scale=params['regularization'])

    # preprocessing
    image_shape = (1, image_resolution, image_resolution, 3)
    images = tf.get_variable("images", image_shape, initializer=init, regularizer=reg)

    scales = [1 + (i - 5) / 50. for i in range(11)]
    angles = list(range(-10, 11)) + 5 * [0]

    if preproc is True:
        images = xforms.pad(images, pad_amount=12)
        images = xforms.jitter(images, jitter_amount=8)
        images = xforms.random_scale(images, scales)
        images = xforms.random_rotate(images, angles)
        images = xforms.jitter(images, jitter_amount=4)

    # get features for a given layer from a given model
    tensor_name = params.get('tensor_name', None)
    layer = model_fn(images, layer_name=layer_name, tensor_name=tensor_name, **model_kwargs)

    # initialize all variables except for 'images'
    sess = tf.Session()

    # extract specified channel from conv or fc layer
    if unit_index is None:
        if len(layer.get_shape().as_list()) == 4:
            target = layer[0, :, :, params['channel']]
        else:
            target = layer[0, params['channel']]
    else:
        unit_row, unit_col = unit_index
        if len(layer.get_shape().as_list()) == 4:
            target = layer[0, unit_row, unit_col, params['channel']]
        else:
            print("Warning: both unit index and channel were provided for an FC layer, which conflicts (channels _are_ units). Going with channel instead of index") 
            target = layer[0, params['channel']]

    # set up loss function
    if loss is None:
        total_reg = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    else:
        tv_loss = total_variation_loss(images)
        total_reg = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) + tv_loss
    loss_tensor = tf.negative(tf.reduce_mean(target)) + total_reg

    # set up optimizer
    lr_tensor = tf.constant(params['learning_rate'])

    # restrict trainable variables to the image itself
    train_vars = [
        var for var in tf.trainable_variables() if 'images' in var.name
    ]
    train_op = tf.train.AdamOptimizer(lr_tensor).minimize(loss_tensor, var_list=train_vars)

    # initialize session and all variables, restore model weights
    # sess.run(tf.initialize_variables([images]))
    sess.run(tf.global_variables_initializer())

    all_variables = tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES)
    temp_saver = tf.train.Saver(
        var_list=[v for v in all_variables if "images" not in v.name and "beta" not in v.name]
    )
    temp_saver.restore(sess, checkpoint_path)

    ## Main Loop
    for i in range(params['steps']):
        sess.run(train_op)

    final_image = sess.run(images)
    return norm_image(final_image.squeeze())
