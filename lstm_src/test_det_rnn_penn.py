import os
import cv2
import sys
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


def main(gpu, prefix, steps, mem_frac):
  image_size = int(prefix.split('image_size=')[1].split('_')[0])
  num_layer = int(prefix.split('num_layer=')[1].split('_')[0])
  lstm_units = int(prefix.split('lstm_units=')[1].split('_')[0])
  seen_step = int(prefix.split('seen_step=')[1].split('_')[0])
  fut_step = steps
  keep_prob = float(prefix.split('keep_prob=')[1].split('_')[0])
  learning_rate = float(prefix.split('learning_rate=')[1].split('_')[0])

  lm_size = 13
  input_size = lm_size * 2
  num_class = 8

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

  samples_dir = './results/pose/' + prefix
  models_dir = './models/' + prefix

  data_path = './datasets/PennAction/'
  testfiles = open(data_path + 'test_subset_list.txt', 'r').readlines()
  alldata = []
  for i in xrange(len(testfiles)):
    vid_path = testfiles[i].split()[0]
    tks = vid_path.split('frames')
    alldata.append(np.load(data_path + '/hourglass/' + tks[1][1:] + '.npz'))

  with tf.device('/gpu:%d' % gpu):
    lstm = DET_LSTM(
        1,
        input_size,
        layers,
        seen_step,
        fut_step,
        keep_prob,
        None,
        learning_rate,
        mode='test')

  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=mem_frac)
  with tf.Session(
      config=tf.ConfigProto(
          allow_soft_placement=True,
          log_device_placement=False,
          gpu_options=gpu_options)) as sess:
    sess.run(tf.global_variables_initializer())

    loaded, _ = lstm.load(sess, models_dir)
    if loaded:
      print("[*] Load SUCCESS")
    else:
      print("[!] Load failed...")
      os._exit(0)

    for i in xrange(len(alldata)):
      print 'Testing: ' + str(i) + '/' + str(len(testfiles))
      pose_batch = np.zeros((1, seen_step, input_size), dtype='float32')
      mask_batch = np.zeros((1, seen_step, lm_size), dtype='float32')
      act_batch = np.zeros((1, num_class), dtype='int32')
      ff = alldata[i]
      posey = transform(ff['y'][:seen_step, :])
      posex = transform(ff['x'][:seen_step, :])
      visib = ff['visibility'][:seen_step]

      pose_batch[0] = np.concatenate((posex, posey), axis=1)
      mask_batch[0] = visib
      act_batch[0, class_dict[str(ff['action'][0])]] = 1
      act = class_dict.keys()[class_dict.values().index(act_batch[0].argmax())]

      res_path = samples_dir + '/{0:05d}'.format(i) + '_' + act
      if not os.path.exists(res_path):
        os.makedirs(res_path)

      output = lstm.predict(sess, pose_batch, mask_batch)
      np.save(res_path + '/joints.npy', inverse_transform(output) * image_size)
      for stp in range(np.min([ff['nframes'][0,0], seen_step + fut_step])):
        pre = output[0, stp, :2 * lm_size]
        posex, posey = (pre[:lm_size], pre[lm_size:])
        if stp < seen_step:
          visib = mask_batch[0, stp]
        else:
          visib = np.ones(posex.shape)
        sample = np.repeat(
            visualize_lm(posex, posey, visib, lines, image_size)[:, :, None],
            [3],
            axis=2)
        sample = draw_frame((sample * 255).astype('uint8'), stp < seen_step)
        cv2.imwrite(res_path + '/{0:05d}'.format(stp) + '.png', sample)

      gif_save_path = res_path + '/pred_' + act + '.gif'

      cmd1 = 'rm ' + gif_save_path
      cmd2 = 'ffmpeg -f image2 -framerate 7 -i %s %s' % (
          res_path + '/%05d.png', gif_save_path)
      cmd3 = 'rm ' + res_path + '/*png'
      system(cmd1)
      system(cmd2)
      system(cmd3)
  print 'Done.'


if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument(
      "--gpu", type=int, dest="gpu", required=True, help="GPU device id")
  parser.add_argument(
      "--prefix", type=str, dest='prefix', required=True, help="Model to test")
  parser.add_argument(
      "--steps",
      type=int,
      dest="steps",
      required=True,
      help="Number of stesp into the future")
  parser.add_argument(
      "--mem_frac",
      type=float,
      dest="mem_frac",
      default=0.1,
      help="GPU memory fraction to take up")
  args = parser.parse_args()
  main(**vars(args))
