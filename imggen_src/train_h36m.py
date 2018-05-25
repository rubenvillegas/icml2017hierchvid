import cv2
import sys
import time
import imageio

import tensorflow as tf
import scipy.misc as sm
import numpy as np
import scipy.io as sio

from model_analogy import IMGGEN
from utils import *
from os import listdir, makedirs, system
from cython.parallel import *
from cython import boundscheck, wraparound
from joblib import Parallel, delayed
from os.path import exists
from argparse import ArgumentParser
from pylab import *


def main(lr, image_size, batch_size, layer, alpha, beta, gamma, steps,
         num_iter, gpu):
  data_path = './datasets/Human3.6M/'
  f = open(data_path + 'train_list.txt', 'r')
  trainfiles = f.readlines()
  margin = 0.3
  updateD = True
  updateG = True
  prefix = ('HUMAN3.6M_ANALOGY_imgsize=' + str(image_size) + '_layer=' +
            str(layer) + '_alpha=' + str(alpha) + '_beta=' + str(beta) +
            '_gamma=' + str(gamma) + '_lr=' + str(lr))
  print('\n' + prefix + '\n')

  checkpoint_dir = './models/' + prefix + '/'
  samples_dir = './samples/' + prefix + '/'
  summary_dir = './logs/' + prefix + '/'

  if not exists(checkpoint_dir):
    makedirs(checkpoint_dir)
  if not exists(samples_dir):
    makedirs(samples_dir)

  with tf.device('/gpu:%d' % gpu[0]):
    model = IMGGEN(
        image_size=image_size,
        batch_size=batch_size,
        layer=layer,
        alpha=alpha,
        beta=beta,
        n_joints=32,
        checkpoint_dir=checkpoint_dir)
    d_optim = tf.train.AdamOptimizer(
        lr, beta1=0.5).minimize(
            model.d_loss, var_list=model.dis_vars)
    g_optim = tf.train.AdamOptimizer(
        lr, beta1=0.5).minimize(
            model.L + gamma * model.g_loss, var_list=model.g_vars)

  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
  with tf.Session(
      config=tf.ConfigProto(
          allow_soft_placement=True,
          log_device_placement=False,
          gpu_options=gpu_options)) as sess:
    tf.global_variables_initializer().run()

    loaded, model_name = model.load(sess, checkpoint_dir)
    if loaded:
      print(" [*] Load SUCCESS")
      iters = int(model_name.split("-")[-1])
    else:
      print(" [!] Load failed...")
      iters = 0

    g_sum  = tf.summary.merge([model.L_img_sum, model.L_feat_sum, model.L_sum,
                               model.g_loss_sum])
    d_sum  = tf.summary.merge([model.d_loss_real_sum, model.d_loss_sum,
                               model.d_loss_fake_sum])
    writer = tf.summary.FileWriter(summary_dir, tf.get_default_graph())

    counter = iters + 1
    start_time = time.time()

    with Parallel(n_jobs=batch_size) as parallel:
      while iters < num_iter:
        mini_batches = get_minibatches_idx(
            len(trainfiles), batch_size, shuffle=True)
        for _, batchidx in mini_batches:
          if len(batchidx) == batch_size:
            seq_batch = np.zeros(
                (batch_size, image_size, image_size, 2, 3), dtype='float32')
            pose_batch = np.zeros(
                (batch_size, image_size, image_size, 2, 32), dtype='float32')

            t0 = time.time()
            tsteps = np.repeat(np.array([steps]), batch_size, axis=0)
            tfiles = np.array(trainfiles)[batchidx]
            shapes = np.repeat(np.array([image_size]), batch_size, axis=0)
            output = parallel(delayed(load_h36m_data)(f, img_sze, s)\
                                                 for f,img_sze,s in zip(tfiles,
                                                                        shapes,
                                                                        tsteps))
            for i in xrange(batch_size):
              seq_batch[i] = output[i][0]
              pose_batch[i] = output[i][1]
            print 'Took: ' + str(time.time() - t0)

            xt = seq_batch[:, :, :, 0, :]
            xtpn = seq_batch[:, :, :, 1, :]
            pt = pose_batch[:, :, :, 0, :]
            ptpn = pose_batch[:, :, :, 1, :]

            feed_dict = {
                model.xt_: xt,
                model.xtpn_: xtpn,
                model.pt_: pt,
                model.ptpn_: ptpn
            }
            if updateG:
              print 'Updating G'
              _, summary_str = sess.run([g_optim, g_sum], feed_dict=feed_dict)
            if updateD:
              print 'Updating D'
              _, summary_str = sess.run([d_optim, d_sum], feed_dict=feed_dict)

            writer.add_summary(summary_str, counter)
            errD_fake = model.d_loss_fake.eval(feed_dict)
            errD_real = model.d_loss_real.eval(feed_dict)
            errG = model.g_loss.eval(feed_dict)

            if errD_fake < margin or errD_real < margin:
              updateD = False
            if errD_fake > (1. - margin) or errD_real > (1. - margin):
              updateG = False
            if not updateD and not updateG:
              updateD = True
              updateG = True

            counter += 1

            print(
                "Iters: [%2d] time: %4.4f, \n\t\t d_loss: %.8f, g_loss: %.8f" %
                (iters, time.time() - start_time, errD_fake + errD_real, errG))

            if np.mod(counter, 100) == 1:
              samples = sess.run([model.G], feed_dict=feed_dict)[0]
              ptpn_rgb = 2 * ptpn.max(axis=-1)[...,None] - 1
              ptpn_rgb = np.concatenate([ptpn_rgb, ptpn_rgb, ptpn_rgb], axis=-1)
              samples = np.concatenate(
                  (samples[:,:,:,0,:], ptpn_rgb, xtpn), axis=0
              )
              print("Saving sample ...")
              save_images(samples[:, :, :, ::-1], [8, 8],
                          samples_dir + 'train_%s.png' % (iters))
            if np.mod(counter, 500) == 2:
              model.save(sess, checkpoint_dir, counter)

            iters += 1


if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument(
      "--lr", type=float, dest="lr", default=0.0001, help="Base Learning Rate")
  parser.add_argument(
      "--image_size",
      type=int,
      dest="image_size",
      default=128,
      help="Pre-trained model")
  parser.add_argument(
      "--batch_size",
      type=int,
      dest="batch_size",
      default=8,
      help="Mini-batch size")
  parser.add_argument(
      "--layer", type=int, dest="layer", default=3, help="hourglass layer")
  parser.add_argument(
      "--alpha",
      type=float,
      dest="alpha",
      default=1.,
      help="weight for alexnet feat-sim")
  parser.add_argument(
      "--beta",
      type=float,
      dest="beta",
      default=1.,
      help="weight for hourglass feat-sim")
  parser.add_argument(
      "--gamma",
      type=float,
      dest="gamma",
      default=1.,
      help="weight for adversarial g_loss")
  parser.add_argument(
      "--steps",
      type=int,
      dest="steps",
      default=1000,
      help="Number of initial frames")
  parser.add_argument(
      "--num_iter",
      type=int,
      dest="num_iter",
      default=100000,
      help="Number of iterations")
  parser.add_argument(
      "--gpu",
      type=int,
      nargs='+',
      dest="gpu",
      required=True,
      help="GPU device id")

  args = parser.parse_args()
  main(**vars(args))
