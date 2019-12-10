# import packages
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import os
import argparse
import ipdb
import sys
import pprint

import tensorflow as tf
import numpy as np

sys.path.append('../models/')
from facenet.src.models.inception_resnet_v1 import inception_resnet_v1
from models_facenet import *

import transformations as xforms
from optimization import get_optimal_image

# set paths
META_PATH = "/share/kalanit/Projects/Dawn/CS229/models/facenet/facenet_checkpoint/model-20180402-114759.meta"
CKPT_PATH = "/share/kalanit/Projects/Dawn/CS229/models/facenet/facenet_checkpoint/model-20180402-114759.ckpt-275"
SAVE_PATH = "/share/kalanit/Projects/Dawn/CS229/figures"

def main():
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = "1"

	layer_params = {
		'Conv2d_1a_3x3': {
		    'channel': 0,
		    'learning_rate': 1e-3,
		    'regularization': 0.0001,
		    'steps': 1000,
		    'tensor_name': 'Conv2d_1a_3x3',
		},
		'Conv2d_2a_3x3': {
		    'channel': 0,
		    'learning_rate': 1e-3,
		    'regularization': 0.0001,
		    'steps': 1000,
		    'tensor_name': 'Conv2d_2a_3x3',
		},
		'Conv2d_2b_3x3': {
		    'channel': 0,
		    'learning_rate': 1e-2,
		    'regularization': 0.0001,
		    'steps': 1000,
		    'tensor_name': 'Conv2d_2b_3x3',
		},
		'Conv2d_3b_1x1': {
		    'channel': 0,
		    'learning_rate': 1e-3,
		    'regularization': 0.0001,
		    'steps': 1000,
		    'tensor_name': 'Conv2d_3b_1x1',
		},
		'Conv2d_4a_3x3': {
		    'channel': 0,
		    'learning_rate': 1e-3,
		    'regularization': 0.0001,
		    'steps': 1000,
		    'tensor_name': 'Conv2d_4a_3x3',
		},
		'Mixed_6a': {
		    'channel': 0,
		    'learning_rate': 1e-3,
		    'regularization': 0.0001,
		    'steps': 1000,
		    'tensor_name': 'Mixed_6a',
		}
		,
		'Mixed_6b': {
		    'channel': 0,
		    'learning_rate': 1e-3,
		    'regularization': 0.0001,
		    'steps': 1000,
		    'tensor_name': 'Mixed_6b',
		}
	    }

	keys = [
		'Conv2d_1a_3x3',
		'Conv2d_2a_3x3',
		'Conv2d_2b_3x3',
		'Conv2d_3b_1x1',
		'Conv2d_4a_3x3',
		'Mixed_6a',
		'Mized_6b'
	]

	facenet_kwargs = {
		'train': False
	}

	loss = 'TV'

	fig, axes = plt.subplots(figsize=(25, 5), nrows=1, ncols=6)
	for layer_name, ax in zip(keys, axes.ravel()):
		layer_dict = layer_params[layer_name]
		optimal_image = get_optimal_image(
		    facenet_no_fc_wrapper,
		    facenet_kwargs,
		    CKPT_PATH,
		    layer_dict,
		    loss,
		    layer_name=layer_name,
		    meta_path=META_PATH,
		)
		ax.imshow(optimal_image)
		ax.axis('off')
		ax.set_title(layer_name)

	save_path = "%s/facenet_testing_norotate_lowlr.png" % SAVE_PATH
	plt.savefig(save_path, dpi=200)

	#inputs = tf.convert_to_tensor(np.random.rand(5, 128, 128, 3))
	#net, endpoints = inception_resnet_v1(
	#	inputs, is_training=False)
	#pprint.pprint(net)
	#pprint.pprint(endpoints)

	#saver = tf.train.import_meta_graph(META_PATH)
        #sess = tf.Session()
        #saver.restore(sess, CKPT_PATH)

	#sess.run(tf.global_variables_initializer())
	#sess.run(tf.local_variables_initializer())
	#conv2d_4b_3x3 = sess.run(endpoints['Conv2d_4b_3x3'])

	#print(conv2d_4b_3x3.shape)
	#print(conv2d_4b_3x3[0, 10, 10, :])
		

if __name__ == "__main__":
	main()
