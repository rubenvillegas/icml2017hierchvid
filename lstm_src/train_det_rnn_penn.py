import os
import cv2
import sys
import ipdb
import time
import socket

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import scipy.misc as sm
import numpy as np
import scipy.io as sio
from os import listdir, makedirs, system
from argparse import ArgumentParser
from utils import *
from det_lstm import DET_LSTM


def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  img = np.zeros((h * size[0], w * size[1]))

  for idx, image in enumerate(images):
    i = idx % size[1]
    j = idx / size[1]
    img[j * h:j * h + h, i * w:i * w + w] = image

  return img


def transform(input_):
  return 2 * input_ - 1.


def inverse_transform(input_):
  return (input_ + 1.) / 2.


def imsave(images, size, path):
  return sm.imsave(path, merge(images, size))


def visualize_lm(posex, posey, visib, lines, image_size):
  posey = inverse_transform(posey) * image_size
  posex = inverse_transform(posex) * image_size
  cpose = np.zeros((image_size, image_size, 48))
  for j in xrange(12):
    if (visib[lines[j][0]] and visib[lines[j][1]] and
        visib[lines[j][2]] and visib[lines[j][3]]):
      interp_x = np.linspace((posex[lines[j][0]] + posex[lines[j][1]]) / 2,
                             (posex[lines[j][2]] + posex[lines[j][3]]) / 2, 4,
                             True)
      interp_y = np.linspace((posey[lines[j][0]] + posey[lines[j][1]]) / 2,
                             (posey[lines[j][2]] + posey[lines[j][3]]) / 2, 4,
                             True)
      for k in xrange(4):
        gmask = gauss2D_mask(
            (interp_y[k], interp_x[k]), (image_size, image_size), sigma=8.)
        cpose[:, :, j * 4 + k] = gmask / gmask.max()
    else:
      if visib[lines[j][0]] and visib[lines[j][1]]:
        point_x = (posex[lines[j][0]] + posex[lines[j][1]]) / 2
        point_y = (posey[lines[j][0]] + posey[lines[j][1]]) / 2
        gmask = gauss2D_mask(
            (point_y, point_x), (image_size, image_size), sigma=8.)
        cpose[:, :, j * 4] = gmask / gmask.max()
      if visib[lines[j][2]] and visib[lines[j][3]]:
        point_x = (posex[lines[j][2]] + posex[lines[j][3]]) / 2
        point_y = (posey[lines[j][2]] + posey[lines[j][3]]) / 2
        gmask = gauss2D_mask(
            (point_y, point_x), (image_size, image_size), sigma=8.)
        cpose[:, :, (j + 1) * 4 - 1] = gmask / gmask.max()

  return np.amax(cpose, axis=2)


def main(gpu, image_size, batch_size, num_layer, lstm_units, seen_step,
         fut_step, mem_frac, keep_prob, learning_rate):

  lm_size = 13
  input_size = lm_size * 2
  num_class = 8

  prefix = 'PENNACTION_DET_LSTM'

  for kk, vv in locals().iteritems():
    if kk != 'prefix' and kk != 'mem_frac' and kk != 'gpu':
      prefix += '_' + kk + '=' + str(vv)

  layers = []
  for i in range(num_layer):
    layers.append(lstm_units)

  lines = [[0, 0, 1, 2], [1, 1, 2, 2], [1, 1, 3, 3], [3, 3, 5, 5],
           [2, 2, 4, 4], [4, 4, 6, 6], [1, 2, 7, 8], [7, 7, 8, 8],
           [7, 7, 9, 9], [9, 9, 11, 11], [8, 8, 10, 10], [10, 10, 12, 12]]

  class_dict = {
      'baseball_pitch': 0,
      'baseball_swing': 1,
      'clean_and_jerk': 2,
      'golf_swing': 3,
      'jumping_jacks': 4,
      'jump_rope': 5,
      'tennis_forehand': 6,
      'tennis_serve': 7
  }

  samples_dir = './samples/' + prefix
  models_dir = './models/' + prefix
  logs_dir = './logs/' + prefix

  data_path = './datasets/PennAction/'
  trainfiles = open(data_path + 'train_subset_list.txt',
                    'r').readlines()
  alldata = []
  for i in xrange(len(trainfiles)):
    vid_path = trainfiles[i].split()[0]
    tks = vid_path.split('frames')
    tdata = np.load(data_path + 'labels/'+ tks[1][1:] + '.npz')
    data = {}
    for kk, vv in tdata.iteritems():
      data[kk] = vv
    data['x'] = data['x'] / (1.0 * data['bbox'][0, 3] - data['bbox'][0, 1])
    data['y'] = data['y'] / (1.0 * data['bbox'][0, 2] - data['bbox'][0, 0])
    alldata.append(data)

  with tf.device('/gpu:%d' % gpu):
    lstm = DET_LSTM(batch_size, input_size, layers, seen_step, fut_step,
                    keep_prob, logs_dir, learning_rate)

  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=mem_frac)
  with tf.Session(
      config=tf.ConfigProto(
          allow_soft_placement=True,
          log_device_placement=False,
          gpu_options=gpu_options)) as sess:
    sess.run(tf.global_variables_initializer())

    loaded, model_name = lstm.load(sess, models_dir)
    if loaded:
      print("[*] Load SUCCESS")
      step = int(model_name.split("-")[-1])
    else:
      print("[!] Load failed...")
      step = 0

    total_steps = round(600000 * 16 / batch_size)
    del_list = None

    while step < total_steps:
      mini_batches, del_list = get_minibatches_idx(
          len(trainfiles),
          batch_size,
          shuffle=True,
          min_frame=None,
          trainfiles=trainfiles,
          del_list=del_list)
      for _, batchidx in mini_batches:
        start_time = time.time()
        if len(batchidx) == batch_size:
          pose_batch = np.zeros(
              (batch_size, seen_step + fut_step, input_size), dtype='float32')
          mask_batch = np.zeros(
              (batch_size, seen_step + fut_step, lm_size), dtype='float32')
          act_batch = np.zeros((batch_size, num_class), dtype='int32')
          for i in xrange(batch_size):
            ff = alldata[batchidx[i]]
            high = ff['nframes'] - fut_step - seen_step + 1
            if ff['nframes'] < fut_step + seen_step:
              stidx = 0
            else:
              stidx = np.random.randint(
                  low=0, high=ff['nframes'] - fut_step - seen_step + 1)

            posey = transform(ff['y'][stidx:stidx + seen_step + fut_step, :])
            posex = transform(ff['x'][stidx:stidx + seen_step + fut_step, :])
            visib = ff['visibility'][stidx:stidx + seen_step + fut_step]

            if posey.shape[0] < fut_step + seen_step:
              n_missing = fut_step + seen_step - posey.shape[0]
              posey = np.concatenate(
                  (posey, np.tile(posey[-1], (n_missing, 1))), axis=0)
              posex = np.concatenate(
                  (posex, np.tile(posex[-1], (n_missing, 1))), axis=0)
              visib = np.concatenate(
                  (visib, np.tile(visib[-1], (n_missing, 1))), axis=0)

            pose_batch[i] = np.concatenate((posex, posey), axis=1)
            mask_batch[i] = visib
            act_batch[i, class_dict[str(ff['action'][0])]] = 1
            lbl = act_batch[i].argmax()

          mid_time = time.time()

          err = lstm.train(
              sess, pose_batch, mask_batch, step, save_logs=True)
          if step % 100 == 0:
            output = lstm.predict(sess, pose_batch, mask_batch)
            samples = None
            for idx in range(1):
              for stp in range(seen_step + fut_step):
                pre = output[idx, stp, :2 * lm_size]
                posex, posey, visib = (pre[:lm_size], pre[lm_size:],
                                       np.ones(mask_batch[idx, stp, :].shape))
                act = class_dict.keys()[
                    class_dict.values().index(act_batch[idx].argmax())]
                visib = np.ones(posex.shape)
                sample = visualize_lm(posex, posey, visib, lines, image_size)
                sample = sample.reshape((1, image_size, image_size))

                samples = sample if samples == None else np.concatenate(
                    [samples, sample], axis=0)

            if not os.path.exists(samples_dir):
              os.makedirs(samples_dir)

            img_save_path = samples_dir + '/{0:07d}'.format(
                step) + '_' + act + '.png'
            imsave(samples, [1, seen_step + fut_step], img_save_path)

          print('step=%d/%d, loss=%.12f,  time=%.2f+%.2f' % (
                step, total_steps, err, mid_time - start_time,
                time.time() - mid_time))

          if step >= 10000 and step % 10000 == 0:
            lstm.save(sess, models_dir, lstm.global_step)

          step = step + 1

    lstm.save(sess, models_dir, lstm.global_step)


if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument(
      "--gpu", type=int, dest="gpu", required=True, help="GPU device id")
  parser.add_argument(
      "--image_size",
      type=int,
      default=128,
      dest="image_size",
      help="Spatial size of image")
  parser.add_argument(
      "--batch_size",
      type=int,
      default=256,
      dest="batch_size",
      help="Batch size for training")
  parser.add_argument(
      "--num_layer",
      type=int,
      default=1,
      dest="num_layer",
      help="Number of hidden layers for LSTM")
  parser.add_argument(
      "--lstm_units",
      type=int,
      default=1024,
      dest="lstm_units",
      help="Number of hidden units for LSTM")
  parser.add_argument(
      "--seen_step",
      type=int,
      default=10,
      dest="seen_step",
      help="Number of seen steps")
  parser.add_argument(
      "--fut_step",
      type=int,
      default=32,
      dest="fut_step",
      help="Number of steps into future")
  parser.add_argument(
      "--mem_frac",
      type=float,
      default=0.4,
      dest="mem_frac",
      help="GPU memory fraction to take up")
  parser.add_argument(
      "--keep_prob",
      type=float,
      default=1.0,
      dest="keep_prob",
      help="Keep probability for dropout")
  parser.add_argument(
      "--learning_rate",
      type=float,
      default=0.001,
      dest="learning_rate",
      help="Keep probability for dropout")
  args = parser.parse_args()
  main(**vars(args))
