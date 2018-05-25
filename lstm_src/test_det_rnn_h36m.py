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


def visualize_lm(posex, posey, image_size):
  posey = inverse_transform(posey) * image_size
  posex = inverse_transform(posex) * image_size
  cpose = np.zeros((image_size, image_size, 32))
  for j in xrange(32):
    gmask = gauss2D_mask(
        (posey[j], posex[j]), (image_size, image_size), sigma=8.)
    cpose[:, :, j] = gmask / gmask.max()

  return np.amax(cpose, axis=2)


def main(gpu, prefix, steps, mem_frac):
  image_size = int(prefix.split('image_size=')[1].split('_')[0])
  num_layer = int(prefix.split('num_layer=')[1].split('_')[0])
  lstm_units = int(prefix.split('lstm_units=')[1].split('_')[0])
  seen_step = int(prefix.split('seen_step=')[1].split('_')[0])
  fut_step = steps
  fskip = int(prefix.split('fskip=')[1].split('_')[0])
  keep_prob = float(prefix.split('keep_prob=')[1].split('_')[0])
  learning_rate = float(prefix.split('learning_rate=')[1].split('_')[0])

  lm_size = 32
  input_size = lm_size * 2

  layers = []
  for i in range(num_layer):
    layers.append(lstm_units)

  class_dict = {
      'walkdog': 0,
      'purchases': 1,
      'waiting': 2,
      'eating': 3,
      'sitting': 4,
      'photo': 5,
      'discussion': 6,
      'greeting': 7,
      'walking': 8,
      'phoning': 9,
      'posing': 10,
      'walktogether': 11,
      'directions': 12,
      'smoking': 13,
      'sittingdown': 14
  }

  num_class = len(class_dict.keys())

  samples_dir = './results/pose/' + prefix
  models_dir = './models/' + prefix

  data_path = './datasets/Human3.6M/'
  testfiles = open(data_path + 'test_list_pose_hg.txt', 'r').readlines()
  alldata = []

  unique_list = []
  for i in xrange(len(testfiles)):
    vid_path = testfiles[i].split('\n')[0]
    data = {}
    tdata = np.load(vid_path)
    for kk, vv in tdata.iteritems():
      data[kk] = vv
    data['all_posex'] = data['all_posex'] / (
        1.0 * data['box'][2] - data['box'][0])
    data['all_posey'] = data['all_posey'] / (
        1.0 * data['box'][3] - data['box'][1])
    class_name = vid_path.split('/')[-1].split()[0].split('.')[0].lower()
    if class_name == 'walkingdog':
      class_name = 'walkdog'
    if class_name == 'takingphoto':
      class_name = 'photo'
    data['action'] = class_name
    if class_name in class_dict.keys() and not class_name in unique_list:
      alldata.append(data)

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
      print 'Testing: ' + str(i) + '/' + str(len(alldata))
      pose_batch = np.zeros((1, seen_step, input_size), dtype='float32')
      mask_batch = np.zeros((1, seen_step, lm_size), dtype='float32')
      act_batch = np.zeros((1, num_class), dtype='int32')
      ff = alldata[i]
      stidx = ff['all_posey'].shape[0] / 2
      endidx = stidx + fskip * seen_step
      posey = transform(ff['all_posey'][stidx:endidx:fskip, :])
      posex = transform(ff['all_posex'][stidx:endidx:fskip, :])

      pose_batch[0] = np.concatenate((posex, posey), axis=1)
      mask_batch[0] = np.ones(mask_batch[0].shape)
      act_batch[0, class_dict[str(ff['action'])]] = 1
      act = class_dict.keys()[class_dict.values().index(act_batch[0].argmax())]

      res_path = samples_dir + '/{0:05d}'.format(i) + '_' + act
      if not os.path.exists(res_path):
        os.makedirs(res_path)

      output = lstm.predict(sess, pose_batch, mask_batch)
      np.save(res_path + '/joints.npy', inverse_transform(output) * image_size)
      for stp in range(seen_step + fut_step):
        pre = output[0, stp, :2 * lm_size]
        posex, posey, visib = (pre[:lm_size], pre[lm_size:],
                               np.ones(mask_batch[0, 0, :].shape))
        visib = np.ones(posex.shape)
        sample = np.repeat(
            visualize_lm(posex, posey, image_size)[:, :, None], [3], axis=2)
        cv2.imwrite(
            res_path + '/{0:05d}'.format(stp) + '.png',
            draw_frame((sample * 255).astype('uint8'), stp < seen_step))

      gif_save_path = res_path + '/pred_' + act + '.gif'

      cmd1 = 'rm ' + gif_save_path
      cmd2 = 'ffmpeg -f image2 -framerate 7 -i %s %s' % (
          res_path + '/%05d.png', gif_save_path)
      cmd3 = 'rm ' + res_path + '/*png'
      system(cmd1)
      system(cmd2)
      #system(cmd3)
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
