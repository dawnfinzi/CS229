"""
Optimize one image each for single channels in Alexnet pre RELU
(Currently mostly for testing other parts of the pipeline)
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

    layer_params = {
        'conv3': {
            'channel': 0,
            'learning_rate': 0.05,
            'regularization': 1e-4,
            'steps': 2048,
            'tensor_name': 'conv_2',
            #'unit_index': (4,4),
            'loss': 'TV',
            'loss_lambda': 0.5,
        }
    }
    keys = [
        'conv3'
    ]

    alexnet_kwargs = {
        'train': False
    }

    # setup optional inputs 
    preproc =True

    layer_dict = layer_params[keys[0]]
    optimal_image, loss_list = get_optimal_image(
            alexnet_no_fc_wrapper,
            alexnet_kwargs,
            CKPT_PATH,
            layer_dict,
            preproc,
            layer_name=None,
        )
    
    plt.imshow(optimal_image)
    save_path = "%s/testing_refactor2.png" % SAVE_PATH
    plt.savefig(save_path, dpi=200)

if __name__ == "__main__":
    # Parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu", dest="gpu", type=str, default="1", help="Which gpu to run this on"
    )

    FLAGS, _ = parser.parse_known_args()
    main()
