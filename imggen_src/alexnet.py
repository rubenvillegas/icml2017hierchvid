"""Implementation from https://www.cs.toronto.edu/~guerzhoy/tf_alexnet/"""
import tensorflow as tf

from utils import *
from ops import relu


def alexnet(image):
  net_data = np.load("./perceptual_models/alexnet/bvlc_alexnet.npy").item()
  k_h = 11
  k_w = 11
  c_o = 96
  s_h = 4
  s_w = 4
  conv1W = tf.Variable(net_data["conv1"][0])
  conv1b = tf.Variable(net_data["conv1"][1])
  conv1 = relu(
      conv(
          image,
          conv1W,
          conv1b,
          k_h,
          k_w,
          c_o,
          s_h,
          s_w,
          padding="SAME",
          group=1))
  radius = 2
  alpha = 2e-05
  beta = 0.75
  bias = 1.0
  lrn1 = tf.nn.local_response_normalization(
      conv1, depth_radius=radius, alpha=alpha, beta=beta, bias=bias)
  k_h = 3
  k_w = 3
  s_h = 2
  s_w = 2
  padding = 'VALID'
  maxpool1 = tf.nn.max_pool(
      lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

  k_h = 5
  k_w = 5
  c_o = 256
  s_h = 1
  s_w = 1
  group = 2
  conv2W = tf.Variable(net_data["conv2"][0])
  conv2b = tf.Variable(net_data["conv2"][1])
  conv2 = relu(
      conv(
          maxpool1,
          conv2W,
          conv2b,
          k_h,
          k_w,
          c_o,
          s_h,
          s_w,
          padding="SAME",
          group=group))
  radius = 2
  alpha = 2e-05
  beta = 0.75
  bias = 1.0
  lrn2 = tf.nn.local_response_normalization(
      conv2, depth_radius=radius, alpha=alpha, beta=beta, bias=bias)
  k_h = 3
  k_w = 3
  s_h = 2
  s_w = 2
  padding = 'VALID'
  maxpool2 = tf.nn.max_pool(
      lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

  k_h = 3
  k_w = 3
  c_o = 384
  s_h = 1
  s_w = 1
  group = 1
  conv3W = tf.Variable(net_data["conv3"][0])
  conv3b = tf.Variable(net_data["conv3"][1])
  conv3 = relu(
      conv(
          maxpool2,
          conv3W,
          conv3b,
          k_h,
          k_w,
          c_o,
          s_h,
          s_w,
          padding="SAME",
          group=group))

  k_h = 3
  k_w = 3
  c_o = 384
  s_h = 1
  s_w = 1
  group = 2
  conv4W = tf.Variable(net_data["conv4"][0])
  conv4b = tf.Variable(net_data["conv4"][1])
  conv4 = relu(
      conv(
          conv3,
          conv4W,
          conv4b,
          k_h,
          k_w,
          c_o,
          s_h,
          s_w,
          padding="SAME",
          group=group))

  k_h = 3
  k_w = 3
  c_o = 256
  s_h = 1
  s_w = 1
  group = 2
  conv5W = tf.Variable(net_data["conv5"][0])
  conv5b = tf.Variable(net_data["conv5"][1])
  conv5 = relu(
      conv(
          conv4,
          conv5W,
          conv5b,
          k_h,
          k_w,
          c_o,
          s_h,
          s_w,
          padding="SAME",
          group=group))

  return conv5


def conv(input_,
         kernel,
         biases,
         k_h,
         k_w,
         c_o,
         s_h,
         s_w,
         padding="VALID",
         group=1):
  """From https://github.com/ethereon/caffe-tensorflow
  """
  c_i = input_.get_shape()[-1]
  assert c_i % group == 0
  assert c_o % group == 0
  convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)

  if group == 1:
    conv = convolve(input_, kernel)
  else:
    input_groups = tf.split(axis=3, num_or_size_splits=group, value=input_)
    kernel_groups = tf.split(axis=3, num_or_size_splits=group, value=kernel)
    output_groups = [
        convolve(i, k) for i, k in zip(input_groups, kernel_groups)
    ]
    conv = tf.concat(axis=3, values=output_groups)
  return tf.reshape(
      tf.nn.bias_add(conv, biases), [-1] + conv.get_shape().as_list()[1:])
