import os
import cv2
import sys
import time
import ssim

import tensorflow as tf
import scipy.misc as sm
import scipy.io as sio
import numpy as np
import skimage.measure as measure

from model_analogy import IMGGEN
from utils import *
from os import listdir, makedirs, system
from os.path import exists
from argparse import ArgumentParser
from pylab import *
from skimage.draw import line_aa
from PIL import Image
from PIL import ImageDraw


def main(imggen_prefix, lstm_prefix, image_size, batch_size, fut_step, gpu):
  data_path = './datasets/PennAction/'
  f = open(data_path + 'test_subset_list.txt', 'r')
  testfiles = f.readlines()
  seen_step = 10
  checkpoint_dir = './models/' + imggen_prefix + '/'

  lines = [[0, 0, 1, 2], [1, 1, 2, 2], [1, 1, 3, 3], [3, 3, 5, 5],
           [2, 2, 4, 4], [4, 4, 6, 6], [1, 2, 7, 8], [7, 7, 8, 8],
           [7, 7, 9, 9], [9, 9, 11, 11], [8, 8, 10, 10], [10, 10, 12, 12]]

  with tf.device('/gpu:%d' % gpu[0]):
    model = IMGGEN(
        image_size=image_size,
        batch_size=batch_size,
        is_train=False,
        checkpoint_dir=checkpoint_dir)

  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
  with tf.Session(
      config=tf.ConfigProto(
          allow_soft_placement=True,
          log_device_placement=False,
          gpu_options=gpu_options)) as sess:
    tf.global_variables_initializer().run()
    start_time = time.time()

    loaded, _ = model.load(sess, checkpoint_dir)
    if loaded:
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed... exitting")
      sys.exit(0)

    pred_path = './results/pose/' + lstm_prefix + '/'

    for i in xrange(len(testfiles)):
      vid_path = testfiles[i].split()[0]
      vid_imgs = sorted(
          [f for f in listdir(vid_path) if f.endswith('cropped.png')])
      print ' Video ' + str(i) + '/' + str(len(testfiles))
      tks = vid_path.split('frames')
      ff = np.load(tks[0] + 'labels/' + tks[1][1:] + '.npz')
      bbox = ff['bbox']
      pose_folder = '{0:05}'.format(i) + '_' + ff['action'][0]
      folder_name = ff['action'][0] + '_' + vid_path.split('/')[-1]
      pred = np.load(pred_path + pose_folder + '/joints.npy')
      ff = {}
      ff['x'] = pred[0, :, :13]
      ff['y'] = pred[0, :, 13:]
      ff['visibility'] = 1.0 - (
          (ff['x'] == 64) * (ff['y'] == 64)).astype('int32')
      ff['x'] = ff['x'] / 128. * (bbox[0, 2] - bbox[0, 0])
      ff['y'] = ff['y'] / 128. * (bbox[0, 3] - bbox[0, 1])
      savedir = ('./results/images/PennAction/' + imggen_prefix +
                 '/' + folder_name)
      if not os.path.exists(savedir):
        os.makedirs(savedir)

      seq_batch = np.zeros(
          (batch_size, image_size, image_size, seen_step + fut_step, 3),
          dtype='float32')
      pose_batch = np.zeros(
          (batch_size, image_size, image_size, seen_step + fut_step, 48),
          dtype='float32')

      steps = np.min([seen_step + fut_step, len(vid_imgs)])
      for t in range(steps):
        img = cv2.imread(vid_path + '/' +
                         vid_imgs[np.min([t, len(vid_imgs) - 1])])
        cpose = np.zeros((img.shape[0], img.shape[1], 48))
        posey = ff['y'][t]
        posex = ff['x'][t]
        visib = ff['visibility'][t]

        for k in xrange(12):
          if (visib[lines[k][0]] and visib[lines[k][1]] and visib[lines[k][2]]
              and visib[lines[k][3]]):
            interp_x = np.linspace(
                (posex[lines[k][0]] + posex[lines[k][1]]) / 2.0,
                (posex[lines[k][2]] + posex[lines[k][3]]) / 2.0, 4, True)
            interp_y = np.linspace(
                (posey[lines[k][0]] + posey[lines[k][1]]) / 2.0,
                (posey[lines[k][2]] + posey[lines[k][3]]) / 2.0, 4, True)
            for m in xrange(4):
              gmask = gauss2D_mask(
                  (interp_y[m], interp_x[m]), img.shape[:2], sigma=8.)
              cpose[:, :, k * 4 + m] = gmask / gmask.max()
          else:
            if visib[lines[k][0]] and visib[lines[k][1]]:
              point_x = (posex[lines[k][0]] + posex[lines[k][1]]) / 2.0
              point_y = (posey[lines[k][0]] + posey[lines[k][1]]) / 2.0
              gmask = gauss2D_mask((point_y, point_x), img.shape[:2], sigma=8.)
              cpose[:, :, k * 4] = gmask / gmask.max()
            if visib[lines[k][2]] and visib[lines[k][3]]:
              point_x = (posex[lines[k][2]] + posex[lines[k][3]]) / 2.0
              point_y = (posey[lines[k][2]] + posey[lines[k][3]]) / 2.0
              gmask = gauss2D_mask((point_y, point_x), img.shape[:2], sigma=8.)
              cpose[:, :, (k + 1) * 4 - 1] = gmask / gmask.max()

        img = cv2.resize(img, (image_size, image_size))
        cpose0 = cv2.resize(cpose, (image_size, image_size))
        seq_batch[0, :, :, t] = transform(img)
        pose_batch[0, :, :, t] = cpose0

      true_data = seq_batch.copy()
      pred_data = np.zeros(true_data.shape, dtype='float32')
      xt = seq_batch[:, :, :, seen_step - 1]
      pt = pose_batch[:, :, :, seen_step - 1]
      for t in xrange(steps - seen_step):
        ptpn = pose_batch[:, :, :, seen_step + t]
        feed_dict = {model.xt_: xt, model.pt_: pt, model.ptpn_: ptpn}
        pred = sess.run(model.G, feed_dict=feed_dict)
        pred_data[:, :, :, t] = pred[:, :, :, 0].copy()

      pred_data = np.concatenate(
          (seq_batch[:, :, :, :seen_step], pred_data), axis=3)
      xt_in = (inverse_transform(xt[0, :, :, 0]) * 255).astype('uint8')
      cv2.imwrite(savedir + '/input_img.png', xt_in)
      pp_in = cv2.cvtColor(
          pose_batch[0, :, :, seen_step - 1].max(axis=-1) * 255.,
          cv2.COLOR_GRAY2RGB)
      cv2.imwrite(savedir + '/input_pose.png', pp_in)
      for t in xrange(steps):
        pred = (inverse_transform(pred_data[0, :, :, t]) * 255).astype('uint8')
        target = (inverse_transform(true_data[0, :, :, t]) * 255).astype('uint8')
        pp = cv2.cvtColor(
            pose_batch[0, :, :, t].max(axis=-1) * 255., cv2.COLOR_GRAY2RGB)
        pred = draw_frame(pred, t < seen_step)
        target = draw_frame(target, t < seen_step)
        pp = draw_frame(pp, t < seen_step)
        cv2.imwrite(savedir + '/gt_' + '{0:04d}'.format(t) + '.png', target)
        cv2.imwrite(savedir + '/ours_' + '{0:04d}'.format(t) + '.png', pred)
        cv2.imwrite(savedir + '/pose_' + '{0:04d}'.format(t) + '.png', pp)

      cmd1 = 'rm ' + savedir + '/ours.gif'
      cmd2 = ('ffmpeg -f image2 -framerate 7 -i ' + savedir + '/ours_%04d.png '
              + savedir + '/ours.gif')
      cmd3 = 'rm ' + savedir + '/ours*.png'
      system(cmd1)
      system(cmd2)
      #system(cmd3);

      cmd1 = 'rm ' + savedir + '/gt.gif'
      cmd2 = ('ffmpeg -f image2 -framerate 7 -i ' + savedir + '/gt_%04d.png ' +
              savedir + '/gt.gif')
      cmd3 = 'rm ' + savedir + '/gt*.png'
      system(cmd1)
      system(cmd2)
      #system(cmd3);

      cmd1 = 'rm ' + savedir + '/pose.gif'
      cmd2 = ('ffmpeg -f image2 -framerate 7 -i ' + savedir + '/pose_%04d.png '
              + savedir + '/pose.gif')
      cmd3 = 'rm ' + savedir + '/pose*.png'
      system(cmd1)
      system(cmd2)
      #system(cmd3);
    print 'Done.'


if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument(
      "--imggen_prefix",
      type=str,
      dest="imggen_prefix",
      required=True,
      help="Prefix for log/snapshot")
  parser.add_argument(
      "--lstm_prefix",
      type=str,
      dest="lstm_prefix",
      required=True,
      help="Prefix for log/snapshot")
  parser.add_argument(
      "--image_size",
      type=int,
      dest="image_size",
      default=128,
      help="Image dimensions")
  parser.add_argument(
      "--batch_size",
      type=int,
      dest="batch_size",
      default=1,
      help="Mini-batch size")
  parser.add_argument(
      "--fut_step",
      type=int,
      dest="fut_step",
      default=64,
      help="Number of steps into the future")
  parser.add_argument(
      "--gpu",
      type=int,
      nargs='+',
      dest="gpu",
      required=True,
      help="GPU device id")

  args = parser.parse_args()
  main(**vars(args))
