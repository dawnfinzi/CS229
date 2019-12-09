"""
Optimize one image each for single channels in Alexnet pre RELU
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
from optimization import get_optimal_image

# set paths
CKPT_PATH = "/share/kalanit/Projects/Dawn/CS229/models/checkpoints/alexnet/model.ckpt-115000"
alexnet_checkpoint_path = "/share/kalanit/Projects/Dawn/CS229/models/checkpoints/alexnet/model.ckpt-115000"
SAVE_PATH = "/share/kalanit/Projects/Dawn/CS229/figures"

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
    keys = [
        'conv1',
        'conv2',
        'conv3',
        'conv4',
        'conv5',
    ]

    alexnet_kwargs = {
        'train': False
    }

    loss = None

    fig, axes = plt.subplots(figsize=(25, 5), nrows=1, ncols=5)
    for layer_name, ax in zip(keys, axes.ravel()):
        layer_dict = layer_params[layer_name]
        optimal_image = get_optimal_image(
            alexnet_no_fc_wrapper,
            alexnet_kwargs,
            alexnet_checkpoint_path,
            layer_dict,
            loss,
            layer_name=None,
        )
        ax.imshow(optimal_image)
        ax.axis('off')
        ax.set_title(layer_name)

    save_path = "%s/alexnet_finalized_noTVloss.png" % SAVE_PATH
    plt.savefig(save_path, dpi=200)

if __name__ == "__main__":
    # Parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu", dest="gpu", type=str, default="1", help="Which gpu to run this on"
    )

    FLAGS, _ = parser.parse_known_args()
    main()
