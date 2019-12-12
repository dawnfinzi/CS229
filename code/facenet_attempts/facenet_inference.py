# import packages
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import os
import argparse
import ipdb
import sys
from pprint import pprint

import tensorflow as tf
import numpy as np

sys.path.append('../models/')
from facenet.src.models.inception_resnet_v1 import inception_resnet_v1
from facenet.tmp.nn4 import inference
from facenet.src.facenet import load_model
from models_facenet import *

import transformations as xforms
from optimization import get_optimal_image

import importlib

# set paths
META_PATH = "/share/kalanit/Projects/Dawn/CS229/models/facenet/facenet_checkpoint/model-20180402-114759.meta"
CKPT_PATH = "/share/kalanit/Projects/Dawn/CS229/models/facenet/facenet_checkpoint/model-20180402-114759.ckpt-275"
model_file = "/share/kalanit/Projects/Dawn/CS229/models/facenet/facenet_checkpoint/"

SAVE_PATH = "/share/kalanit/Projects/Dawn/CS229/figures"

channel = 0
params = {
	'learning_rate': 1e-3,
	'regularization': 1e-3,
	'steps': 100,
	}

def norm_image(x):
	return (x - np.min(x))/np.ptp(x)

def sanity_check(sess):
	weights_tensor = tf.get_default_graph().get_tensor_by_name("InceptionResnetV1/Conv2d_1a_3x3/weights:0")
	weights = sess.run(weights_tensor)

	target  = 0.059823785
	run_weight = weights[0, 0, 0, 0]
	assert(np.isclose(target, run_weight))


def main():
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = "1"

	sess = tf.Session()

	#saver = tf.train.import_meta_graph(META_PATH)
	#saver.restore(sess, CKPT_PATH)
	#sanity_check(sess)	

	# Initialize an image with random noise
	image_init = tf.random_uniform_initializer(
		minval=0,
		maxval=1,
	)
	# add regularization
	image_reg = tf.contrib.layers.l2_regularizer(
		scale=params['regularization']
	)
	# now define the image tesnor using the initializer and the regularizer
	#t_input = tf.placeholder(np.float32, shape=(128,128,3), name='input') # define the input tensor

	image_shape = (1, 128, 128, 3)
	images = tf.get_variable(
		"images",
		image_shape,
		initializer=image_init,
		regularizer=image_reg
	)

	network = inference(images,1.0, phase_train=True, weight_decay=0.0)
	pprint(network)
	ipdb.set_trace()
	layer = network[1]['conv1']
	
	# Create a saver for restoring variables
	#saver = tf.train.Saver(tf.global_variables())
	# Restore the parameters
	#saver.restore(sess, model_file)
	#saver = tf.train.import_meta_graph(META_PATH)
	#saver.restore(sess, CKPT_PATH)

	# ipdb.set_trace()
	#sess.run(tf.global_variables_initializer())
	#saver.restore(sess, CKPT_PATH)
	#layer_name = 'InceptionResnetV1/Repeat_2/block8_3/Conv2d_1x1/Conv2D'
	#layer = tf.get_default_graph().get_tensor_by_name('%s:0' % layer_name)

	#aver = tf.train.import_meta_graph(META_PATH)
	#saver.restore(sess, CKPT_PATH)
	# find the placeholders
	#graph = tf.get_default_graph()
	#placeholders = [op for op in graph.get_operations() if op.type == "Placeholder"]
	#ipdb.set_trace()

	#names = [n.name for n in tf.get_default_graph().as_graph_def().node]
	#bs_names = [name for name in names if "batch_size" in name]
	#pprint(bs_names)
	# ipdb.set_trace()


	total_reg = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
	loss = tf.negative(layer[:,:,:,channel]) + total_reg
	pprint(loss)

	vars_to_train = [var for var in tf.trainable_variables() if "images:0" == var.name]
	optimizer = tf.train.AdamOptimizer(params['learning_rate'])
	train_op = optimizer.minimize(loss, var_list=vars_to_train)

	loss_list = list()
	image_list = list()

	sess.run(tf.global_variables_initializer())
	saver = tf.train.import_meta_graph(META_PATH)
	saver.restore(sess, CKPT_PATH)

	for step in range(params['steps']):
		loss_list.append(sess.run(loss))
		image_list.append(norm_image(sess.run(images)))
		sess.run(train_op)
		
	sanity_check(sess)

	print(image_list.shape)

	# """ # let's get the tensor to optimize from facenet
	# inference(
	# 	images,
	# 	1.0,
	# 	phase_train=True,
	# 	weight_decay=0.0
	# )
	# layers = [op.name for op in tf.get_default_graph().get_operations() if op.type=='Conv2D']
	# #layer = tf.get_default_graph().get_tensor_by_name("InceptionResnetV1/Conv2d_1a_3x3/Conv2D")
	# #pprint(layer)
	
	# feature_nums = {layer: int(T(layer).get_shape()[-1]) for layer in layers}
	# print('Number of layers: %d' % len(layers))
	
	# for layer in sorted(feature_nums.keys()):
	# 	print('%s%d' % ((layer+': ').ljust(40), feature_nums[layer])) """


	# weights_tensor = tf.get_default_graph().get_tensor_by_name("InceptionResnetV1/Conv2d_1a_3x3/weights:0")
	# weights = sess.run(weights_tensor)
	# print(weights.shape)
	# print(weights[0,0,0,0])

	# weights = np.moveaxis(weights, 3, 0)
	# fig, axes = plt.subplots(figsize=(24, 16), nrows=4, ncols=8)
	# for kernel_idx, (kernel, ax) in enumerate(zip(weights, axes.ravel())):
	# 	normed = (kernel - np.min(kernel)) / np.ptp(kernel)
	# 	ax.imshow(normed)
	# 	ax.axis('off')
	# 	ax.set_title(kernel_idx)

	# save_path = "%s/facenet_weights2.png" % SAVE_PATH
	# plt.savefig(save_path, dpi=200)

if __name__ == "__main__":
	main()
