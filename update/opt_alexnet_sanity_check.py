"""
Optimize an image for a unit in AlexNet's first convolutional layer 
and compare to the actual weights in order to check that the 
optimization procedure is working
"""

# import packages
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import os
import argparse

import ipdb
import numpy as np
import random
import tensorflow as tf

from models_alexnet import *
import transformations as xforms
from opt_utils import get_optimal_image
from utils import norm_image

# set paths
CKPT_PATH = "/share/kalanit/Projects/Dawn/CS229/models/checkpoints/alexnet/model.ckpt-115000"
SAVE_PATH = "/share/kalanit/Projects/Dawn/CS229/figures"

def main():
    print("Using GPU %s" % FLAGS.gpu)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

    unit_x = 8
    unit_y = 8
    #pick a random channel
    channel = random.randrange((96))

    layer_params = {
        'conv1': {
            'channel': channel,
            'learning_rate': 0.05,
            'regularization': 1e-4,
            'steps': 500,
            'tensor_name': 'conv',
            'unit_index': (unit_x,unit_y),
        }
    }
    keys = [
        'conv1'
    ]

    alexnet_kwargs = {
        'train': False
    }

    # setup optional inputs 
    preproc = False
    loss = None

    ## Run imageopt
    layer_dict = layer_params['conv1']
    optimal_image, loss_list = get_optimal_image(
            alexnet_no_fc_wrapper,
            alexnet_kwargs,
            CKPT_PATH,
            layer_dict,
            preproc,
            layer_name=None,
        )

    ## Get weights back
    tf.reset_default_graph()
    init = tf.random_uniform_initializer(minval=0, maxval=1) #initialize random noise
    image_shape = (1, 128, 128, 3)
    images = tf.get_variable("images", image_shape, initializer=init)
    #use random noise image to get alexnet graph back
    model = alexnet_no_fc_wrapper(images, tensor_name = 'conv', train=False)
    sess = tf.Session()
    all_variables = tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES)
    temp_saver = tf.train.Saver(
        var_list=[v for v in all_variables if "images" not in v.name and "beta" not in v.name]
    )
    temp_saver.restore(sess, CKPT_PATH) #restore checkpoint weights
    weights_tensor = tf.get_default_graph().get_tensor_by_name("conv1/weights:0")
    weights = sess.run(weights_tensor)
    
    ## Plot
    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(17, 5), nrows=1, ncols=3)
    title = "AlexNet sanity check: Channel %d" % (channel)
    fig.suptitle(title)

    true_filter = norm_image(weights[:,:,:,channel])
    ax1.imshow(true_filter)
    ax1.set_title("True filter")

    ax2.imshow(optimal_image)
    ax2.set_title("Optimized image")

    ax3.plot(loss_list, c='k', linewidth=5)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.set_ylabel('Loss')
    ax3.set_xlabel('Steps')
    ax3.set_title("Loss over iterations")

    if loss is None:
        save_path = "%s/alexnet_sanity_check_channel%d.png" % (SAVE_PATH, channel)
    else:
        save_path = "%s/alexnet_sanity_check_channel%d_with%sloss.png" % (SAVE_PATH, channel,loss)
    plt.savefig(save_path, dpi=200)

if __name__ == "__main__":
    # Parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu", dest="gpu", type=str, default="1", help="Which gpu to run this on"
    )

    FLAGS, _ = parser.parse_known_args()
    main()
