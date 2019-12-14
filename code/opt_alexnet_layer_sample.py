"""
Optimize one image each for a channel in each layer in Alexnet pre RELU
"""

# import packages
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import os
import argparse

import ipdb
import numpy as np
import tensorflow as tf

from models_alexnet import *
import transformations as xforms
from opt_utils import get_optimal_image

# set paths
CKPT_PATH = "/share/kalanit/Projects/Dawn/CS229/models/checkpoints/alexnet/model.ckpt-115000"
SAVE_PATH = "/share/kalanit/Projects/Dawn/CS229/figures"

def main():
    print("Using GPU %s" % FLAGS.gpu)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

    fig, axes = plt.subplots(figsize=(25, 5), nrows=1, ncols=5)

    loss = 'TV'
    preproc = True

    params = {
        'channel':0,
        'learning_rate': .05,
        'regularization': 0.0001,
        'steps': 1000, # 2048,
        'loss': loss,
        #'loss_lambda': 1.0,
    }

    preproc_params = {
        'pad': 20, #12,
        'scale': True,
        'rotate': True,
        'pre_jitter': 8,
        'post_jitter': 8,
    }

    layer_names = [
        'conv1',
        'conv2',
        'conv3',
        'conv4',
        'conv5',
    ]

    tensor_names = [
        'conv',
        'conv_1',
        'conv_2',
        'conv_3',
        'conv_4',
    ]

    alexnet_kwargs = {
        'train': False
    }

    for tensor_name, ax in zip(tensor_names, axes.ravel()):
        print("Processing %s" % tensor_name)
        params['tensor_name'] = tensor_name
        optimal_image, loss_list = get_optimal_image(
            alexnet_no_fc_wrapper,
            alexnet_kwargs,
            CKPT_PATH,
            params,
            preproc,
            preproc_params, 
            layer_name=None,
        )

        ax.imshow(optimal_image)
        ax.axis('off')

    if preproc is True:
        if loss is None:
            save_path = "%s/alexnet_sample_preproc.png" % (SAVE_PATH)
        else: 
            save_path = "%s/alexnet_sample_preproc_%sloss.png" % (SAVE_PATH, loss)
    else:
        if loss is None:
            save_path = "%s/alexnet_sample_nopreproc.png" % (SAVE_PATH)
        else: 
            save_path = "%s/alexnet_sample_nopreproc_%sloss.png" % (SAVE_PATH, loss)

    plt.savefig(save_path, dpi=300)

if __name__ == "__main__":
    # Parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu", dest="gpu", type=str, default="1", help="Which gpu to run this on"
    )

    FLAGS, _ = parser.parse_known_args()
    main()