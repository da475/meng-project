
# Author: Deepak, Jaydev
# Deep-net for mnist

# A deep MNIST classifier using convolutional layers.
# reference: https://www.tensorflow.org/get_started/mnist/pros

# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
import pickle
import pydicom
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None
"""
  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer.
  with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)
NOTE:
FEATURES_FC_LAYER is no of neurons in fc layer, candidate for HP
TODO:
use macro
use shape
"""

# working numbers on mac

FEATURES_1st_LAYER  = 16

IMAGE_HEIGHT_HALF   = 50
IMAGE_WIDTH_HALF    = 50
IMAGE_SLICES_HALF   = 10

IMAGE_HEIGHT        = IMAGE_HEIGHT_HALF * 2
IMAGE_WIDTH         = IMAGE_WIDTH_HALF * 2
IMAGE_SLICES        = IMAGE_SLICES_HALF * 2

FEATURES_FC_LAYER   = 1024

def deepnn(x):
  """deepnn builds the graph for a deep net for classifying digits.
  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.
  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, IMAGE_SLICES, IMAGE_HEIGHT, IMAGE_HEIGHT, 1])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([5, 5, 5, 1, FEATURES_1st_LAYER])
    b_conv1 = bias_variable([FEATURES_1st_LAYER])
    h_conv1 = tf.nn.relu(conv3d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
  with tf.name_scope('pool1'):
    h_pool1 = max_pool3d_2x2x2(h_conv1)

  # Fully connected layer 1 -- after 2 round of downsampling, our IMAGE_HEIGHTxIMAGE_HEIGHT image
  # is down to 7x7x64 feature maps -- maps this to FEATURES_FC_LAYER features.
  with tf.name_scope('fc1'):
    current_feat = IMAGE_SLICES_HALF * IMAGE_HEIGHT_HALF * IMAGE_WIDTH_HALF * FEATURES_1st_LAYER
    W_fc1 = weight_variable([ current_feat, FEATURES_FC_LAYER])
    b_fc1 = bias_variable([FEATURES_FC_LAYER])

    h_pool2_flat = tf.reshape(h_pool1, [-1, current_feat])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  """
    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    with tf.name_scope('dropout'):
      keep_prob = tf.placeholder(tf.float32)
      h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
  """

  keep_prob = tf.placeholder(tf.float32)
  # Map the FEATURES_FC_LAYER features to 2 classes
  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([FEATURES_FC_LAYER, 2])
    b_fc2 = bias_variable([2])

    y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

  return y_conv, keep_prob


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def conv3d(x, W):
  """conv3d returns a 3d convolution layer with full stride."""
  return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def max_pool3d_2x2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1],
                        strides=[1, 2, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def main(_):

  size_dataset = 4

  ########### DICOM ###########

  """
  f = pydicom.read_file('0002.DCM')
  single_image = (f.pixel_array).astype(np.float32)
  single_image = single_image[0:IMAGE_SLICES, 0:IMAGE_HEIGHT, 0:IMAGE_HEIGHT]
  data_array_images = np.array([single_image for i in range(0, size_dataset)])
  print (data_array_images.shape, ' is the shape')
  """

  data_array_images = np.load('image.npy')
  data_array_labels = np.load('label.npy')
  #single_image = single_image[0:IMAGE_SLICES, 0:IMAGE_HEIGHT, 0:IMAGE_HEIGHT]
  #data_array_images = np.array([single_image for i in range(0, size_dataset)])
  #data_array_labels = [1,0,0,1]
  print (data_array_images.shape, ' is the shape')
  print (data_array_labels.shape, ' is the shape')


  # load the training dataset from the pickle file
  #f = open('dataset_labels.pkl', 'rb')
  #data_array_labels = pickle.load(f)
  #f.close()

  #f = open('dataset_train_images.pkl', 'rb')
  #data_array_images = pickle.load(f)
  #f.close()

  # Create the model
  x_val = tf.placeholder(tf.float32, [None, IMAGE_SLICES, IMAGE_HEIGHT, IMAGE_HEIGHT])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.int64, [None])

  # Build the graph for the deep net
  y_conv, keep_prob = deepnn(x_val)

  with tf.name_scope('loss'):
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(
        labels=y_, logits=y_conv)
  cross_entropy = tf.reduce_mean(cross_entropy)

  with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), y_)
    correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  graph_location = tempfile.mkdtemp()
  print('Saving graph to: %s' % graph_location)
  train_writer = tf.summary.FileWriter(graph_location)
  train_writer.add_graph(tf.get_default_graph())


  with tf.Session() as sess:
    print("before init")
    sess.run(tf.global_variables_initializer())    #todo
    print("after init")
    for i in range(size_dataset):
      print("started iter ", i)
      batch_image = data_array_images
      batch_label = data_array_labels
      print ("x=", x_val.shape, " input=", batch_image.shape)
      input_dict = {x_val: batch_image, y_: batch_label, keep_prob: 0.5}
      train_accuracy = accuracy.eval(feed_dict=input_dict)
      print('step %d, training accuracy %g' % (i, train_accuracy))
      train_step.run(feed_dict=input_dict)

    #print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

if __name__ == '__main__':
  tf.app.run(main=main, argv=[sys.argv[0]])


