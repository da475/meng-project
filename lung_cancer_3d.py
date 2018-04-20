
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


  ########### DICOM ###########

  # this is the numpy array for data images of 4-dimensions
  # it is of the shape: number of images, depth, height, width
  total_images = np.load('81data_points.npy')

  # this is the labels for the above data images
  total_labels = np.load('81labels.npy')

  # take 80% for training, 20% for testing
  num_images = len(total_labels)
  num_training_images = int(num_images * 4 / 5)
  print ('num tr is ', num_training_images)

  # divide the samples
  """
  training_images = total_images[0 : num_training_images-1]
  training_labels = total_labels[0 : num_training_images-1]
  testing_images = total_images[num_training_images : num_images]
  testing_labels = total_labels[num_training_images : num_images]
  """

  training_images = total_images[0 : 10]
  training_labels = total_labels[0 : 10]
  testing_images = total_images[10 : 15]
  testing_labels = total_labels[10 : 15]

  print (training_images.shape, ' is the shape of tr images')
  print (training_labels.shape, ' is the shape of tr labels')
  print (testing_images.shape, ' is the shape of tst images')
  print (testing_labels.shape, ' is the shape of tst labels')

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

  number_of_epochs = 100

  with tf.Session() as sess:
    # start the training
    sess.run(tf.global_variables_initializer())
    print("start the training")
    for i in range(number_of_epochs):
      print("started iter ", i)
      batch_image = training_images
      batch_label = training_labels
      #print ("x=", x_val.shape, " input=", batch_image.shape)
      training_input = {x_val: batch_image, y_: batch_label, keep_prob: 0.5}
      train_accuracy = accuracy.eval(feed_dict = training_input)
      print('step %d, training accuracy %g' % (i, train_accuracy))
      train_step.run(feed_dict=training_input)

    # calculate the testing accuracy
    testing_input = {x_val: testing_images, y_: testing_labels, keep_prob: 0.8}
    test_accuracy = accuracy.eval(feed_dict = testing_input)
    print ('test accu is ', test_accuracy)
    #print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

if __name__ == '__main__':
  tf.app.run(main=main, argv=[sys.argv[0]])



