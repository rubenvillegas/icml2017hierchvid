import os
import cv2
import sys
import time
import ssim
import imageio

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
from joblib import Parallel, delayed
from skimage.draw import line_aa
from PIL import Image
from PIL import ImageDraw


def main(imggen_prefix, lstm_prefix, image_size, batch_size, fut_step, gpu):
  data_path = './datasets/Human3.6M/'
  f = open(data_path + 'test_list_pose.txt', 'r')
  testfiles = f.readlines()
  fskip = 4
  seen_step = 10
  checkpoint_dir = './models/' + imggen_prefix + '/'

  with tf.device('/gpu:%d' % gpu[0]):
    model = IMGGEN(
        image_size=image_size,
        batch_size=batch_size,
        is_train=False,
        checkpoint_dir=checkpoint_dir,
        n_joints=32)

  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
  with tf.Session(
      config=tf.ConfigProto(
          allow_soft_placement=True,
          log_device_placement=False,
          gpu_options=gpu_options)) as sess:
    tf.global_variables_initializer().run()
    counter = 1
    start_time = time.time()

    loaded, _ = model.load(sess, checkpoint_dir)
    if loaded:
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed... exitting")
      sys.exit(0)

    pred_path = './results/pose/' + lstm_prefix + '/'

    for i in xrange(len(testfiles)):
      print ' Video ' + str(i) + '/' + str(len(testfiles))
      vid_path = (testfiles[i].split('MyPoseFeatures')[0] + 'Videos/' +
                  testfiles[i].split('/')[-1].split(' pose')[0].split('.npz')[0])

      vid_imgs = sorted([f for f in listdir(vid_path) if f.endswith('.png')])
      anno_path = (vid_path.split('Videos')[0] + 'MyPoseFeatures/D2_Positions'
                   + vid_path.split('Videos')[1] + '.npz')
      act = testfiles[i].split('/')[-1].split()[0].split('.')[0].lower()
      pose_data = np.load(pred_path + '{0:05}'.format(i) + '_' + act +
                          '/joints.npy')

      box = np.load(anno_path)['box']
      all_posey = (pose_data[:, :, 32:] / 128 * (box[3] - box[1])).round()
      all_posex = (pose_data[:, :, :32] / 128 * (box[2] - box[0])).round()
      path_tks = vid_path.split('/')
      name_tks = path_tks[-1].split(' ')
      if len(name_tks) == 2:
        folder_name = path_tks[-3] + '_' + name_tks[0] + '_' + name_tks[1]
      else:
        folder_name = path_tks[-3] + '_' + name_tks[0]

      savedir = './results/images/Human3.6m/' + imggen_prefix + '/' +folder_name
      if not os.path.exists(savedir):
        os.makedirs(savedir)

      seq_batch = np.zeros(
          (batch_size, image_size, image_size, seen_step + fut_step, 3),
          dtype='float32')
      pose_batch = np.zeros(
          (batch_size, image_size, image_size, seen_step + fut_step, 32),
          dtype='float32')
      bshape = [box[2] - box[0], box[3] - box[1]]
      stidx = len(vid_imgs) / 2
      endidx = stidx + fskip * (seen_step + fut_step)
      vid_imgs = vid_imgs[stidx:endidx:fskip]

      for t in xrange(len(vid_imgs)):
        posey = all_posey[0, t, :]
        posex = all_posex[0, t, :]
        img = cv2.imread(vid_path + '/' + vid_imgs[t])
        cpose = np.zeros((bshape[1], bshape[0], 32))
        for j in xrange(32):
          gmask = gauss2D_mask(
              (posey[j], posex[j]), (bshape[1], bshape[0]), sigma=8.)
          cpose[:, :, j] = gmask / gmask.max()
        img = cv2.resize(img, (image_size, image_size))
        cpose = cv2.resize(cpose, (image_size, image_size))
        seq_batch[0, :, :, t] = transform(img)
        pose_batch[0, :, :, t] = cpose

      true_data = seq_batch.copy()
      pred_data = np.zeros(true_data.shape, dtype='float32')
      xt = seq_batch[:, :, :, seen_step - 1]
      pt = pose_batch[:, :, :, seen_step - 1]
      for t in xrange(fut_step):
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
      for t in xrange(seen_step + fut_step):
        pred = (inverse_transform(pred_data[0, :, :, t]) * 255).astype('uint8')
        target = (inverse_transform(true_data[0, :, :, t]) * 255).astype('uint8')
        pp = cv2.cvtColor(
            pose_batch[0, :, :, t].max(axis=-1) * 255., cv2.COLOR_GRAY2RGB)
        pred = draw_frame(pred, t < seen_step)
        target = draw_frame(target, t < seen_step)
        pp = draw_frame(pp, t < seen_step)
        cv2.imwrite(savedir + '/ours_' + '{0:04d}'.format(t) + '.png', pred)
        cv2.imwrite(savedir + '/gt_' + '{0:04d}'.format(t) + '.png', target)
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
      default=128,
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
