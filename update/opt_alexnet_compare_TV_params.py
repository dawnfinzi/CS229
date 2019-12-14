"""
Compare results for different weightings of the TV loss
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

    scalars_to_plot = np.arange(0.0,1.0,.1)
    fig, axes = plt.subplots(figsize=(15, 5), nrows=2, ncols=5)

    params = {
                'channel': 1,
                'learning_rate': .05,
                'regularization': 0.0001,
                'steps': 1000,
                'loss': 'TV',
                'loss_lambda': 0,
            }

    layer_names = [
        'conv3',
        #'conv4',
        #'conv5',
    ]

    tensor_names = [
        'conv_2',
        #'conv_3',
        #'conv_4',
    ]

    alexnet_kwargs = {
        'train': False
    }

    loss = 'TV'
    preproc = True

    for layer_name, tensor_name in zip(layer_names, tensor_names):
        print("Processing %s" % layer_name)
        for loss_lambda, ax in zip(scalars_to_plot, axes.ravel()):
            params['loss_lambda'] = loss_lambda
            print("Processing loss %d" % loss_lambda)
            params['tensor_name'] = tensor_name
            optimal_image, loss_list = get_optimal_image(
                alexnet_no_fc_wrapper,
                alexnet_kwargs,
                CKPT_PATH,
                params,
                preproc,
                layer_name=None,
            )
            ax.imshow(optimal_image)
            ax.set_title(str(round(loss_lambda,2)))
            ax.axis('off')
        
        save_path = "%s/alexnet_%s_TVloss_lambdas.png" % (SAVE_PATH, layer_name)
        plt.savefig(save_path, dpi=300)

if __name__ == "__main__":
    # Parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu", dest="gpu", type=str, default="1", help="Which gpu to run this on"
    )

    FLAGS, _ = parser.parse_known_args()
    main()
