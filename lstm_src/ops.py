import math
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops

from utils import *


def batch_norm(inputs, name, train=True, reuse=False):
  return tf.contrib.layers.batch_norm(
      inputs=inputs, is_training=train, reuse=reuse, scope=name, scale=True)


def conv1d(input_,
           output_dim,
           initializer='xavier',
           k_w=5,
           d_w=1,
           stddev=0.02,
           padding='VALID',
           reuse=False,
           name="conv1d"):
  with tf.variable_scope(name, reuse=reuse):
    if initializer == 'xavier':
      init_type = tf.contrib.layers.xavier_initializer()
    elif initializer == 'normal':
      init_type = tf.truncated_normal_initializer(stddev=stddev)
    else:
      raise Exception('Weight initializer unknown')
    w = tf.get_variable(
        'w', [k_w, input_.get_shape()[-1], output_dim], initializer=init_type)
    conv = tf.nn.conv1d(input_, w, stride=d_w, padding=padding)
    biases = tf.get_variable(
        'biases', [output_dim], initializer=tf.constant_initializer(0.0))
    conv = tf.nn.bias_add(conv, biases)
    return conv


def conv2d(input_,
           output_dim,
           initializer='xavier',
           k_h=5,
           k_w=5,
           d_h=1,
           d_w=1,
           stddev=0.02,
           padding='VALID',
           reuse=False,
           name="conv2d"):
  with tf.variable_scope(name, reuse=reuse):
    if initializer == 'xavier':
      init_type = tf.contrib.layers.xavier_initializer()
    elif initializer == 'normal':
      init_type = tf.truncated_normal_initializer(stddev=stddev)
    else:
      raise Exception('Weight initializer unknown')

    w = tf.get_variable(
        'w', [k_h, k_w, input_.get_shape()[-1], output_dim],
        initializer=init_type)
    conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)

    biases = tf.get_variable(
        'biases', [output_dim], initializer=tf.constant_initializer(0.0))
    conv = tf.nn.bias_add(conv, biases)

    return conv


def conv2da(input_,
            output_dim,
            k_h=5,
            k_w=5,
            d_h=2,
            d_w=2,
            stddev=0.02,
            name="conv2d",
            reuse=False,
            padding='SAME'):
  with tf.variable_scope(name, reuse=reuse):
    w = tf.get_variable(
        'w', [k_h, k_w, input_.get_shape()[-1], output_dim],
        initializer=tf.contrib.layers.xavier_initializer())
    #                             initializer=tf.truncated_normal_initializer(stddev=stddev))
    conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)

    biases = tf.get_variable(
        'biases', [output_dim], initializer=tf.constant_initializer(0.0))
    conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

    return conv


def conv2dp(input_, output_dim, params, d_h=1, d_w=1):
  w = tf.Variable(params[0])
  conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

  biases = tf.Variable(params[1])
  conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
  return conv


def conv3d(input_,
           output_dim,
           initializer='xavier',
           k_h=5,
           k_w=5,
           d_h=2,
           d_w=2,
           stddev=0.02,
           name="conv3d"):
  with tf.variable_scope(name):
    if initializer == 'xavier':
      init_type = tf.contrib.layers.xavier_initializer()
    elif initializer == 'normal':
      init_type = tf.truncated_normal_initializer(stddev=stddev)
    else:
      raise Exception('Weight initializer unknown')

    w = tf.get_variable(
        'w',
        [input_.get_shape()[-2], k_h, k_w,
         input_.get_shape()[-1], output_dim],
        initializer=init)
    conv = tf.nn.conv3d(input_, w, strides=[1, d_h, d_w, 1, 1], padding='SAME')

    biases = tf.get_variable(
        'biases', [output_dim], initializer=tf.constant_initializer(0.0))
    conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

    return conv


def deconv2d(input_,
             output_shape,
             initializer='xavier',
             k_h=5,
             k_w=5,
             d_h=2,
             d_w=2,
             stddev=0.02,
             name="deconv2d",
             with_w=False):
  with tf.variable_scope(name):
    if initializer == 'xavier':
      init_type = tf.contrib.layers.xavier_initializer()
    elif initializer == 'normal':
      init_type = tf.truncated_normal_initializer(stddev=stddev)
    else:
      raise Exception('Weight initializer unknown')

    # filter : [height, width, output_channels, in_channels]
    w = tf.get_variable(
        'w', [k_h, k_h, output_shape[-1],
              input_.get_shape()[-1]],
        initializer=init_type)

    try:
      deconv = tf.nn.conv2d_transpose(
          input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])

    # Support for verisons of TensorFlow before 0.7.0
    except AttributeError:
      deconv = tf.nn.deconv2d(
          input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])

    biases = tf.get_variable(
        'biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
    deconv = tf.nn.bias_add(deconv, biases)

    if with_w:
      return deconv, w, biases
    else:
      return deconv


def lrelu(x, leak=0.2, name="lrelu"):
  with tf.variable_scope(name):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)


def relu(x):
  return tf.nn.relu(x)


def tanh(x):
  return tf.nn.tanh(x)


def shape2d(a):
  """
    a: a int or tuple/list of length 2
    """
  if type(a) == int:
    return [a, a]
  if isinstance(a, (list, tuple)):
    assert len(a) == 2
    return list(a)
  raise RuntimeError("Illegal shape: {}".format(a))


def shape4d(a):
  # for use with tensorflow
  return [1] + shape2d(a) + [1]


def UnPooling2x2ZeroFilled(x):
  out = tf.concat(3, [x, tf.zeros_like(x)])
  out = tf.concat(2, [out, tf.zeros_like(out)])

  sh = x.get_shape().as_list()
  if None not in sh[1:]:
    out_size = [-1, sh[1] * 2, sh[2] * 2, sh[3]]
    return tf.reshape(out, out_size)
  else:
    sh = tf.shape(x)
    return tf.reshape(out, [-1, sh[1] * 2, sh[2] * 2, sh[3]])


def MaxPooling(x, shape, stride=None, padding='VALID'):
  """
    MaxPooling on images.
    :param input: NHWC tensor.
    :param shape: int or [h, w]
    :param stride: int or [h, w]. default to be shape.
    :param padding: 'valid' or 'same'. default to 'valid'
    :returns: NHWC tensor.
    """
  padding = padding.upper()
  shape = shape4d(shape)
  if stride is None:
    stride = shape
  else:
    stride = shape4d(stride)

  return tf.nn.max_pool(x, ksize=shape, strides=stride, padding=padding)


#@layer_register()
def FixedUnPooling(x, shape, unpool_mat=None):
  """
    Unpool the input with a fixed mat to perform kronecker product with.
    :param input: NHWC tensor
    :param shape: int or [h, w]
    :param unpool_mat: a tf/np matrix with size=shape. If None, will use a mat
        with 1 at top-left corner.
    :returns: NHWC tensor
    """
  shape = shape2d(shape)

  # a faster implementation for this special case
  if shape[0] == 2 and shape[1] == 2 and unpool_mat is None:
    return UnPooling2x2ZeroFilled(x)

  input_shape = tf.shape(x)
  if unpool_mat is None:
    mat = np.zeros(shape, dtype='float32')
    mat[0][0] = 1
    unpool_mat = tf.Variable(mat, trainable=False, name='unpool_mat')
  elif isinstance(unpool_mat, np.ndarray):
    unpool_mat = tf.Variable(unpool_mat, trainable=False, name='unpool_mat')
  assert unpool_mat.get_shape().as_list() == list(shape)

  # perform a tensor-matrix kronecker product
  fx = flatten(tf.transpose(x, [0, 3, 1, 2]))
  fx = tf.expand_dims(fx, -1)  # (bchw)x1
  mat = tf.expand_dims(flatten(unpool_mat), 0)  #1x(shxsw)
  prod = tf.matmul(fx, mat)  #(bchw) x(shxsw)
  prod = tf.reshape(
      prod,
      tf.pack([
          -1, input_shape[3], input_shape[1], input_shape[2], shape[0],
          shape[1]
      ]))
  prod = tf.transpose(prod, [0, 2, 4, 3, 5, 1])
  prod = tf.reshape(
      prod,
      tf.pack([
          -1, input_shape[1] * shape[0], input_shape[2] * shape[1],
          input_shape[3]
      ]))
  return prod


def linear(input_,
           output_size,
           name,
           stddev=0.02,
           bias_start=0.0,
           reuse=False,
           with_w=False):
  shape = input_.get_shape().as_list()

  with tf.variable_scope(name, reuse=reuse):
    matrix = tf.get_variable(
        "Matrix", [shape[1], output_size],
        tf.float32,
        tf.random_normal_initializer(stddev=stddev))
    bias = tf.get_variable(
        "bias", [output_size], initializer=tf.constant_initializer(bias_start))
    if with_w:
      return tf.matmul(input_, matrix) + bias, matrix, bias
    else:
      return tf.matmul(input_, matrix) + bias
