import imageio
import random
import math
import cv2
import time
import numpy as np
import scipy.io as sio
import scipy.misc as sm
from os import listdir, makedirs, system


def transform(X):
  return X / 127.5 - 1


def inverse_transform(X):
  return (X + 1.) / 2.


def save_images(images, size, image_path, bw=False, mean=None):
    if bw:
        imsave(images*255, size, image_path)
    else:
        if mean == None:
            return imsave(inverse_transform(images)*255., size, image_path)
        else:
            return imsave(images*255.+mean[None,None,None,:], size, image_path)


def imsave(images, size, path):
    return sm.imsave(path, merge(images, size))


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx / size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img


def get_minibatches_idx(n, minibatch_size, shuffle=False):
  """
  Used to shuffle the dataset at each iteration.
  """

  idx_list = np.arange(n, dtype="int32")

  if shuffle:
    random.shuffle(idx_list)

  minibatches = []
  minibatch_start = 0
  for i in range(n // minibatch_size):
    minibatches.append(
        idx_list[minibatch_start:minibatch_start + minibatch_size])
    minibatch_start += minibatch_size

  if (minibatch_start != n):
    # Make a minibatch out of what is left
    minibatches.append(idx_list[minibatch_start:])

  return zip(range(len(minibatches)), minibatches)


def gauss2D_mask(center, shape, sigma=0.5):
  m, n = [ss - 1 for ss in shape]
  y, x = np.ogrid[0:m + 1, 0:n + 1]
  y = y - center[0]
  x = x - center[1]
  z = x * x + y * y
  h = np.exp(-z / (2. * sigma * sigma))
  sumh = h.sum()
  if sumh != 0:
    h = h / sumh
  return h


def load_penn_data(f_name, data_path, image_size, steps):
  lines = [[0, 0, 1, 2], [1, 1, 2, 2], [1, 1, 3, 3], [3, 3, 5, 5],
           [2, 2, 4, 4], [4, 4, 6, 6], [1, 2, 7, 8], [7, 7, 8, 8],
           [7, 7, 9, 9], [9, 9, 11, 11], [8, 8, 10, 10], [10, 10, 12, 12]]

  rnd_steps = np.random.randint(1, steps)
  vid_path = f_name.split()[0]
  vid_imgs = sorted(
      [f for f in listdir(vid_path) if f.endswith('cropped.png')])
  low = 0
  high = len(vid_imgs) - rnd_steps - 1
  if high < 1:
    rnd_steps = rnd_steps + high
    high = 1
  stidx = np.random.randint(low=0, high=high)
  seq = np.zeros((image_size, image_size, 2, 3), dtype='float32')
  pose = np.zeros((image_size, image_size, 2, 48), dtype='float32')
  for t in xrange(2):
    img = cv2.imread(vid_path + '/' + vid_imgs[stidx + t * rnd_steps])
    cpose = np.zeros((img.shape[0], img.shape[1], 48))
    tks = vid_path.split('frames')
    ff = np.load(tks[0] + 'labels/' + tks[1][1:] + '.npz')
    posey = 1.0 * ff['y'][stidx + t * rnd_steps]
    posex = 1.0 * ff['x'][stidx + t * rnd_steps]
    visib = ff['visibility'][stidx + t * rnd_steps]
    for j in xrange(12):
      if (visib[lines[j][0]] and visib[lines[j][1]] and visib[lines[j][2]]
          and visib[lines[j][3]]):
        interp_x = np.linspace((posex[lines[j][0]] + posex[lines[j][1]]) / 2.0,
                               (posex[lines[j][2]] + posex[lines[j][3]]) / 2.0,
                               4, True)
        interp_y = np.linspace((posey[lines[j][0]] + posey[lines[j][1]]) / 2.0,
                               (posey[lines[j][2]] + posey[lines[j][3]]) / 2.0,
                               4, True)
        for k in xrange(4):
          gmask = gauss2D_mask(
              (interp_y[k], interp_x[k]), img.shape[:2], sigma=8.)
          cpose[:, :, j * 4 + k] = gmask / gmask.max()
      else:
        if visib[lines[j][0]] and visib[lines[j][1]]:
          point_x = (posex[lines[j][0]] + posex[lines[j][1]]) / 2.0
          point_y = (posey[lines[j][0]] + posey[lines[j][1]]) / 2.0
          gmask = gauss2D_mask((point_y, point_x), img.shape[:2], sigma=8.)
          cpose[:, :, j * 4] = gmask / gmask.max()
        if visib[lines[j][2]] and visib[lines[j][3]]:
          point_x = (posex[lines[j][2]] + posex[lines[j][3]]) / 2.0
          point_y = (posey[lines[j][2]] + posey[lines[j][3]]) / 2.0
          gmask = gauss2D_mask((point_y, point_x), img.shape[:2], sigma=8.)
          cpose[:, :, (j + 1) * 4 - 1] = gmask / gmask.max()

    img = cv2.resize(img, (image_size, image_size))
    cpose = cv2.resize(cpose, (image_size, image_size))
    seq[:, :, t] = transform(img)
    pose[:, :, t] = cpose

  return seq, pose


def load_h36m_data(f_name, image_size, steps):
  rnd_steps = np.random.randint(1, steps)
  vid_path = f_name.split('\n')[0].split('.mp4')[0]
  vid_imgs = sorted([f for f in listdir(vid_path) if f.endswith('.png')])
  anno_path = (vid_path.split('Videos')[0] + 'MyPoseFeatures/D2_Positions' +
               vid_path.split('Videos')[1] + '.npz')
  pose_data = np.load(anno_path)
  all_posey = pose_data['all_posey']
  all_posex = pose_data['all_posex']
  box = pose_data['box']

  high = np.min([all_posey.shape[0], len(vid_imgs)]) - rnd_steps - 1
  if high < 1:
    rnd_steps = rnd_steps + high
    high = 1

  stidx = np.random.randint(low=0, high=high)

  seq = np.zeros((image_size, image_size, 2, 3), dtype='float32')
  pose = np.zeros((image_size, image_size, 2, 32), dtype='float32')
  shape = [box[2] - box[0], box[3] - box[1]]

  for t in xrange(2):
    posey = all_posey[stidx + t * rnd_steps, :]
    posex = all_posex[stidx + t * rnd_steps, :]
    img = cv2.imread(vid_path + '/' + vid_imgs[stidx + t * rnd_steps])
    cpose = np.zeros((shape[0], shape[1], 32), dtype='float32')
    for j in xrange(32):
      gmask = gauss2D_mask((posey[j], posex[j]), shape, sigma=8.)
      cpose[:, :, j] = gmask / gmask.max()
    img = cv2.resize(img, (image_size, image_size))
    cpose = cv2.resize(cpose, (image_size, image_size))
    seq[:, :, t] = transform(img)
    pose[:, :, t] = cpose

  return seq, pose


def draw_frame(img, is_input):
  if is_input:
    img[:2, :, 0] = img[:2, :, 2] = 0
    img[:, :2, 0] = img[:, :2, 2] = 0
    img[-2:, :, 0] = img[-2:, :, 2] = 0
    img[:, -2:, 0] = img[:, -2:, 2] = 0
    img[:2, :, 1] = 255
    img[:, :2, 1] = 255
    img[-2:, :, 1] = 255
    img[:, -2:, 1] = 255
  else:
    img[:2, :, 0] = img[:2, :, 1] = 0
    img[:, :2, 0] = img[:, :2, 2] = 0
    img[-2:, :, 0] = img[-2:, :, 1] = 0
    img[:, -2:, 0] = img[:, -2:, 1] = 0
    img[:2, :, 2] = 255
    img[:, :2, 2] = 255
    img[-2:, :, 2] = 255
    img[:, -2:, 2] = 255

  return img
