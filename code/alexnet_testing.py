## Testing image optimization using Alexnet

# import packages
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import os
import argparse

from models_alexnet import *

import ipdb
import numpy as np
import tensorflow as tf
import transformations as xforms

# set paths
CKPT_PATH = "/share/kalanit/Projects/Dawn/CS229/models/checkpoints/alexnet/model.ckpt-115000"
alexnet_checkpoint_path = "/share/kalanit/Projects/Dawn/CS229/models/checkpoints/alexnet/model.ckpt-115000"
SAVE_PATH = "/share/kalanit/Projects/Dawn/CS229/figures"

def norm_image(x):
    return (x - np.min(x))/np.ptp(x)

def main():
    print("Using GPU %s" % FLAGS.gpu)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

    layer_params = {
        'conv1': {
            'channel': 0,
            'learning_rate': 0.05,
            'regularization': 0.0001,
            'steps': 2048,
            'tensor_name': 'conv',
        },
        'conv2': {
            'channel': 0,
            'learning_rate': 0.05,
            'regularization': 0.0001,
            'steps': 2048,
            'tensor_name': 'conv_1',
        },
        'conv3': {
            'channel': 0,
            'learning_rate': 0.05,
            'regularization': 0.0001,
            'steps': 2048,
            'tensor_name': 'conv_2',
        },
        'conv4': {
            'channel': 0,
            'learning_rate': 0.05,
            'regularization': 0.0001,
            'steps': 2048,
            'tensor_name': 'conv_3',
        },
        'conv5': {
            'channel': 6,
            'learning_rate': 0.05,
            'regularization': 0.0001,
            'steps': 2048,
            'tensor_name': 'conv_4',
        },
    }

        #     'channel': 1,
        #     'learning_rate': 30.,
        #     'regularization': 0.0001,
        #     'steps': 250,
        # },
        # 'conv2': {
        #     'channel': 0,
        #     'learning_rate': 30.,
        #     'regularization': 0.0001,
        #     'steps': 250,
        # },
        # 'conv3': {
        #     'channel': 0,
        #     'learning_rate': 30.,
        #     'regularization': 0.0001,
        #     'steps': 250,
        # },
        # 'conv4': {
        #     'channel': 0,
        #     'learning_rate': 30.,
        #     'regularization': 0.0001,
        #     'steps': 250,
        # },
        # 'conv5': {
        #     'channel': 6,
        #     'learning_rate': 30.,
        #     'regularization': 0.0001,
        #     'steps': 250,
        # }
        # ,
        # 'fc6': {
        #     'channel': 0,
        #     'learning_rate': 30.,
        #     'regularization': 0.0001,
        #     'steps': 250,
        # },
        # 'fc7': {
        #     'channel': 0,
        #     'learning_rate': 30.,
        #     'regularization': 0.0001,
        #     'steps': 250,
        # },
        # 'fc8': {
        #     'channel': 0,
        #     'learning_rate': 30.,
        #     'regularization': 0.0001,
        #     'steps': 250,
        # },
    #}

    keys = [
        'conv1',
        'conv2',
        'conv3',
        'conv4',
        'conv5'
        # ,
        # 'fc6',
        # 'fc7',
        # 'fc8',
    ]

    alexnet_kwargs = {
        'train': False
    }

    fig, axes = plt.subplots(figsize=(20, 10), nrows=1, ncols=5)
    for layer_name, ax in zip(keys, axes.ravel()):
        layer_dict = layer_params[layer_name]
        print("Processing %s" % layer_name)
        #ax.imshow(optimization(layer_name, layer_dict))
        opt_image = optimization(
            alexnet_no_fc_wrapper,
            alexnet_kwargs,
            alexnet_checkpoint_path,
            layer_dict,
            layer_name=None,
        )
        ax.imshow(opt_image)
        ax.axis('off')
        ax.set_title(layer_name)

    plt.savefig("%s/alexnet_preproc.png" % SAVE_PATH, dpi=200)

def optimization(
    model_fn,
    model_kwargs,
    checkpoint_path,
    params,
    layer_name=None,
    image_resolution=128,
    unit_index=None
):
    #####REDO THIS!!!
    # set up the model
    tf.reset_default_graph()
    init = tf.random_uniform_initializer(minval=0, maxval=1)
    reg = tf.contrib.layers.l2_regularizer(scale=params['regularization']) #added L2

    image_shape = (1, image_resolution, image_resolution, 3)
    images = tf.get_variable("images", image_shape, initializer=init, regularizer=reg)

    scales = [1 + (i - 5) / 50. for i in range(11)]
    angles = list(range(-10, 11)) + 5 * [0]

    images = xforms.pad(images, pad_amount=12)
    images = xforms.jitter(images, jitter_amount=8)
    images = xforms.random_scale(images, scales)
    images = xforms.random_rotate(images, angles)
    images = xforms.jitter(images, jitter_amount=4)

    # get features for a given layer from a given model
    tensor_name = params.get('tensor_name', None)
    layer = model_fn(images, layer_name=layer_name, tensor_name=tensor_name, **model_kwargs)
    
    # initalize tensorflow session
    #model = alexnet_no_fc(images, train=False)
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
    # instead of gradient ascent do gradient descent using "tf.negative(tf.reduce_mean(channel))"
    loss_tensor = tf.negative(tf.reduce_mean(target)) + tf.reduce_sum(
        tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    )
    #loss_tensor = tf.negative(tf.reduce_mean(channel))
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

    # # restrict trainable variables to the image itself
    # train_vars = [
    #     var for var in tf.trainable_variables() if 'images' in var.name
    # ]

    # # set up optimizer
    # train_op = tf.train.GradientDescentOptimizer(lr_tensor).minimize(loss_tensor, var_list=train_vars)


    # all_variables = tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES)
    # temp_saver = tf.train.Saver(
    #     var_list=[v for v in all_variables if "images" not in v.name]
    # )

    # # get channel
    # layer = model.layers[layer_name]
    # if len(layer.get_shape().as_list()) == 4:
    #     channel = layer[0, :, :, params['channel']]
    # else:
    #     channel = layer[0, params['channel']]

    # # set up loss function
    # # instead of gradient ascent do gradient descent using "tf.negative(tf.reduce_mean(channel))"
    # loss_tensor = tf.negative(tf.reduce_mean(channel)) + tf.reduce_sum(
    #     tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    # )
    # #loss_tensor = tf.negative(tf.reduce_mean(channel))
    # lr_tensor = tf.constant(params['learning_rate'])

    # # restrict trainable variables to the image itself
    # train_vars = [
    #     var for var in tf.trainable_variables() if 'images' in var.name
    # ]

    # # set up optimizer
    # train_op = tf.train.GradientDescentOptimizer(lr_tensor).minimize(loss_tensor, var_list=train_vars)

    # # initialize session and all variables, restore model weights
    # temp_saver.restore(sess, CKPT_PATH)
    # sess.run(tf.global_variables_initializer())
    # #sess.run(tf.initialize_variables([images]))

    # ## Main Loop
    # for i in range(params['steps']):
    #     sess.run(train_op)

    # final_image = sess.run(images)
    # return norm_image(final_image.squeeze())

if __name__ == "__main__":
    # Parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu", dest="gpu", type=str, default="1", help="Which gpu to run this on"
    )

    FLAGS, _ = parser.parse_known_args()
    main()
