import numpy as np
import tensorflow as tf
from ops import *


def convBlock(numIn, numOut, inp, params, idx):
  epsilon = 1e-5
  beta1 = tf.Variable(params[idx])
  idx = idx + 1
  gamma1 = tf.Variable(params[idx])
  idx = idx + 1
  batch_mean1, batch_var1 = tf.nn.moments(inp, [0, 1, 2], name='moments')
  bn1 = tf.nn.batch_norm_with_global_normalization(
      inp,
      batch_mean1,
      batch_var1,
      beta1,
      gamma1,
      epsilon,
      scale_after_normalization=False)
  bn1 = relu(bn1)
  conv1 = conv2dp(bn1, numOut / 2, params[idx:idx + 2])
  idx = idx + 2

  beta2 = tf.Variable(params[idx])
  idx = idx + 1
  gamma2 = tf.Variable(params[idx])
  idx = idx + 1
  batch_mean2, batch_var2 = tf.nn.moments(conv1, [0, 1, 2], name='moments')
  bn2 = tf.nn.batch_norm_with_global_normalization(
      conv1,
      batch_mean2,
      batch_var2,
      beta2,
      gamma2,
      epsilon,
      scale_after_normalization=False)
  bn2 = relu(bn2)
  conv2 = conv2dp(bn2, numOut / 2, params[idx:idx + 2])
  idx = idx + 2

  beta3 = tf.Variable(params[idx])
  idx = idx + 1
  gamma3 = tf.Variable(params[idx])
  idx = idx + 1
  batch_mean3, batch_var3 = tf.nn.moments(conv2, [0, 1, 2], name='moments')
  bn3 = tf.nn.batch_norm_with_global_normalization(
      conv2,
      batch_mean3,
      batch_var3,
      beta3,
      gamma3,
      epsilon,
      scale_after_normalization=False)
  bn3 = relu(bn3)
  conv3 = conv2dp(bn3, numOut, params[idx:idx + 2])
  idx = idx + 2

  return conv3, idx


def skipLayer(numIn, numOut, inp, params, idx):
  if numIn == numOut:
    return inp, idx
  else:
    conv1 = conv2dp(inp, numOut, params[idx:idx + 2])
    idx = idx + 2
    return conv1, idx


def residual(numIn, numOut, inp, params, idx):
  convb, idx = convBlock(numIn, numOut, inp, params, idx)
  skipl, idx = skipLayer(numIn, numOut, inp, params, idx)
  return tf.add(convb, skipl), idx


def hourglass(n, numIn, numOut, inp, params, idx, layers):
  up1, idx = residual(numIn, 256, inp, params, idx)
  up2, idx = residual(256, 256, up1, params, idx)
  up4, idx = residual(256, numOut, up2, params, idx)

  pool1 = MaxPooling(inp, [2, 2])
  low1, idx = residual(numIn, 256, pool1, params, idx)
  low2, idx = residual(256, 256, low1, params, idx)
  low5, idx = residual(256, 256, low2, params, idx)

  if n > 1:
    low6, idx, layers = hourglass(n - 1, 256, numOut, low5, params, idx,
                                  layers)
  else:
    low6, idx = residual(256, numOut, low5, params, idx)

  low7, idx = residual(numOut, numOut, low6, params, idx)
  up5 = tf.image.resize_images(
      low7, [int(low7.get_shape()[1] * 2),
             int(low7.get_shape()[2] * 2)])
  layers.append(tf.add(up4, up5))
  return tf.add(up4, up5), idx, layers


def get_params():
  params = []
  for i in xrange(1, 801):
    p = np.load('./perceptual_models/hourglass/hourglass_weights_' + str(i) +
                '.npy')
    if len(p.shape) == 4:
      p = p.swapaxes(0, 1).swapaxes(0, 2).swapaxes(1, 3)
    params.append(p)
  return params


def hg_forward(inp):
  epsilon = 1e-5
  params = get_params()
  idx = 0
  inp = tf.image.resize_images(inp, [256, 256])
  conv1_ = conv2dp(inp, 64, params[idx:idx + 2], d_h=2, d_w=2)
  idx = idx + 2
  beta1 = tf.Variable(params[idx])
  idx = idx + 1
  gamma1 = tf.Variable(params[idx])
  idx = idx + 1
  batch_mean1, batch_var1 = tf.nn.moments(conv1_, [0, 1, 2], name='moments')
  bn1 = tf.nn.batch_norm_with_global_normalization(
      conv1_,
      batch_mean1,
      batch_var1,
      beta1,
      gamma1,
      epsilon,
      scale_after_normalization=False)
  conv1 = relu(bn1)

  r1, idx = residual(64, 128, conv1, params, idx)
  pool = MaxPooling(r1, [2, 2])
  r4, idx = residual(128, 128, pool, params, idx)
  r5, idx = residual(128, 128, r4, params, idx)
  r6, idx = residual(128, 256, r5, params, idx)
  layers = []
  out, idx, layers = hourglass(4, 256, 512, r6, params, idx, layers)
  return layers
