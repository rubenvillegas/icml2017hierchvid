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


def visualize_lm(posex, posey, image_size):
  posey = inverse_transform(posey) * image_size
  posex = inverse_transform(posex) * image_size
  cpose = np.zeros((image_size, image_size, 32))
  for j in xrange(32):
    gmask = gauss2D_mask(
        (posey[j], posex[j]), (image_size, image_size), sigma=8.)
    cpose[:, :, j] = gmask / gmask.max()

  return np.amax(cpose, axis=2)


def main(gpu, image_size, batch_size, num_layer, lstm_units, seen_step,
         fut_step, mem_frac, keep_prob, learning_rate):

  lm_size = 32
  input_size = lm_size * 2
  fskip = 4

  prefix = 'HUMAN3.6M_DET_LSTM'

  for kk, vv in locals().iteritems():
    if kk != 'prefix' and kk != 'mem_frac' and kk != 'gpu':
      prefix += '_' + kk + '=' + str(vv)

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

  samples_dir = './samples/' + prefix
  models_dir = './models/' + prefix
  logs_dir = './logs/' + prefix

  data_path = './datasets/Human3.6M/'
  trainfiles = open(data_path + 'train_list_pose.txt', 'r').readlines()
  alldata = []
  for i in xrange(len(trainfiles)):
    vid_path = trainfiles[i].split('\n')[0]
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
    if class_name in class_dict.keys():
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
          len(alldata),
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
            nframes = ff['all_posex'].shape[0]
            high = nframes - fskip * (fut_step + seen_step) + 1
            stidx = np.random.randint(low=0, high=high)
            posey = transform(
                ff['all_posey'][stidx:stidx +
                                fskip * (seen_step + fut_step):fskip, :])
            posex = transform(
                ff['all_posex'][stidx:stidx +
                                fskip * (seen_step + fut_step):fskip, :])

            pose_batch[i] = np.concatenate((posex, posey), axis=1)
            mask_batch[i] = np.ones(mask_batch[i].shape)
            act_batch[i, class_dict[str(ff['action'])]] = 1

          mid_time = time.time()

          err = lstm.train(
              sess, pose_batch, mask_batch, step, save_logs=True)

          if step % 100 == 0:
            output = lstm.predict(sess, pose_batch, mask_batch)
            samples = None
            for idx in range(1):
              for stp in range(seen_step + fut_step):
                pre = output[idx, stp, :2 * lm_size]
                posex, posey = (pre[:lm_size], pre[lm_size:])
                act = class_dict.keys()[class_dict.values().index(
                    act_batch[idx].argmax())]
                sample = visualize_lm(posex, posey, image_size)
                sample = sample.reshape((1, image_size, image_size))

                samples = sample if samples is None else np.concatenate(
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
