import random
import math
import cv2
import numpy as np
import pylab as plt
import scipy.io as sio
import scipy.misc as sm
import os
from os import listdir, makedirs, system


def get_minibatches_idx(n,
                        minibatch_size,
                        shuffle=False,
                        min_frame=None,
                        trainfiles=None,
                        del_list=None):
  """
  Used to shuffle the dataset at each iteration.
  """
  idx_list = np.arange(n, dtype="int32")

  if min_frame != None:
    if del_list == None:
      del_list = list()
      for i in idx_list:
        vid_path = trainfiles[i].split()[0]
        length = len([f for f in listdir(vid_path) if f.endswith('.png')])
        if length < min_frame:
          del_list.append(i)
      print('[!] Discarded %d samples from training set!' % len(del_list))
    idx_list = np.delete(idx_list, del_list)

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

  return zip(range(len(minibatches)), minibatches), del_list


def get_minibatches_idx2(n, minibatch_size, shuffle=False):
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


def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  img = np.zeros((h * size[0], w * size[1]))

  for idx, image in enumerate(images):
    i = idx % size[1]
    j = idx / size[1]
    img[j * h:j * h + h, i * w:i * w + w] = image

  return img


def imsave(images, size, path):
  return sm.imsave(path, merge(images, size))


def visualize_lm(posex, posey, visib, vid_path, vid_imgs, stidx, t,
                 image_size):

  lines = [[0, 0, 1, 2], [1, 1, 2, 2], [1, 1, 3, 3], [3, 3, 5, 5],
           [2, 2, 4, 4], [4, 4, 6, 6], [1, 2, 7, 8], [7, 7, 8, 8],
           [7, 7, 9, 9], [9, 9, 11, 11], [8, 8, 10, 10], [10, 10, 12, 12]]

  img = cv2.imread(vid_path + '/' + vid_imgs[stidx])
  visib[posey > 1], visib[posey < 0], visib[posex > 1], visib[
      posex < 0] = 0, 0, 0, 0
  posey = posey * img.shape[0]
  posex = posex * img.shape[1]
  cpose = np.zeros((img.shape[0], img.shape[1], 48))
  for j in range(12):
    if visib[lines[j][0]] and visib[lines[j][1]] and \
       visib[lines[j][2]] and visib[lines[j][3]]:
      interp_x = np.linspace((posex[lines[j][0]] + posex[lines[j][1]]) / 2,
                             (posex[lines[j][2]] + posex[lines[j][3]]) / 2, 4,
                             True)
      interp_y = np.linspace((posey[lines[j][0]] + posey[lines[j][1]]) / 2,
                             (posey[lines[j][2]] + posey[lines[j][3]]) / 2, 4,
                             True)
      for k in range(4):
        gmask = gauss2D_mask(
            (interp_y[k], interp_x[k]), img.shape[:2], sigma=8.)
        cpose[:, :, j * 4 + k] = gmask / gmask.max()
    else:
      if visib[lines[j][0]] and visib[lines[j][1]]:
        point_x = (posex[lines[j][0]] + posex[lines[j][1]]) / 2
        point_y = (posey[lines[j][0]] + posey[lines[j][1]]) / 2
        gmask = gauss2D_mask((point_y, point_x), img.shape[:2], sigma=8.)
        cpose[:, :, j * 4] = gmask / gmask.max()
      if visib[lines[j][2]] and visib[lines[j][3]]:
        point_x = (posex[lines[j][2]] + posex[lines[j][3]]) / 2
        point_y = (posey[lines[j][2]] + posey[lines[j][3]]) / 2
        gmask = gauss2D_mask((point_y, point_x), img.shape[:2], sigma=8.)
        cpose[:, :, (j + 1) * 4 - 1] = gmask / gmask.max()

  cpose = cv2.resize(cpose, (image_size, image_size))
  return np.amax(cpose, axis=2)


def save_test_result(save_path, idx, pred, pose, mask, stidx, seen_step,
                     fut_step, vid_path, fname):
  vid_path = vid_path.split()[0]
  lm_size = 13
  image_size = 128
  samples = None

  fname = fname.split()[0].split('frames')[1][1:]

  save_path_tmp = save_path + '%s/' % fname
  if not os.path.exists(save_path_tmp):
    os.makedirs(save_path_tmp)

  vid_imgs = sorted([f for f in listdir(vid_path) if f.endswith('.png')])
  img = cv2.imread(vid_path + '/' + vid_imgs[0])

  mat = {}
  mat['x'] = pred[:, :lm_size] * img.shape[1]
  mat['y'] = pred[:, lm_size:] * img.shape[0]
  sio.savemat(save_path_tmp + 'joints.mat', mat)

  for typ in range(2):
    for stp in range(fut_step):

      if typ == 0:
        pre = pred[stp, :2 * lm_size]
        posex, posey, visib = (pre[:lm_size], pre[lm_size:],
                               mask[seen_step + stp, :])
      else:
        posex, posey, visib = (pose[seen_step + stp, :lm_size],
                               pose[seen_step + stp, lm_size:],
                               mask[seen_step + stp, :])

      sample = visualize_lm(posex, posey, visib, vid_path, vid_imgs, stidx,
                            stp, image_size)

      file_name = ('pred_' if typ == 0 else 'gt_') + '%04d.png' % stp
      sm.imsave(save_path_tmp + '/' + file_name, sample)

      if typ == 0:
        sample_full = visualize_lm(posex, posey, np.ones_like(visib), vid_path,
                                   vid_imgs, stidx, stp, image_size)
        full_name = 'pred_full_%04d.png' % stp
        sm.imsave(save_path_tmp + '/' + full_name, sample_full)
      sample = sample.reshape((1, image_size, image_size))
      samples = sample if samples is None else np.concatenate(
          [samples, sample], axis=0)

  imsave(samples, [2, fut_step], save_path_tmp + '/' + 'gt_pred.png')

  cmd1 = 'rm ' + save_path_tmp + '/' + 'gt.gif'
  cmd2 = 'ffmpeg -loglevel panic -f image2 -framerate 7 -i '+save_path_tmp\
       +'/gt_%04d.png '+save_path_tmp+'/gt.gif'
  #system(cmd1);
  system(cmd2)

  cmd1 = 'rm ' + save_path_tmp + '/' + 'pred.gif'
  cmd2 = 'ffmpeg -loglevel panic -f image2 -framerate 7 -i '+save_path_tmp\
       +'/pred_%04d.png '+save_path_tmp+'/pred.gif'
  #system(cmd1);
  system(cmd2)

  cmd1 = 'rm ' + save_path_tmp + '/' + 'pred_full.gif'
  cmd2 = 'ffmpeg -loglevel panic -f image2 -framerate 7 -i '+save_path_tmp\
       +'/pred_full_%04d.png '+save_path_tmp+'/pred_full.gif'
  #system(cmd1);
  system(cmd2)

  print('\t%d done.' % idx)


def save_test_result_all(save_path, idx, pred, pose, mask, stidx, seen_step,
                         fut_step, vid_path, fname):
  vid_path = vid_path.split()[0]
  lm_size = 13
  image_size = 128
  samples = None

  fname = fname.split()[0].split('frames')[1][1:]

  save_path_tmp = save_path + '%s/' % fname
  if not os.path.exists(save_path_tmp):
    os.makedirs(save_path_tmp)

  vid_imgs = sorted([f for f in listdir(vid_path) if f.endswith('.png')])
  img = cv2.imread(vid_path + '/' + vid_imgs[0])

  mat = {}
  mat['x'] = pred[:, :lm_size] * img.shape[1]
  mat['y'] = pred[:, lm_size:] * img.shape[0]
  sio.savemat(save_path_tmp + 'joints.mat', mat)

  for typ in range(2):
    for stp in range(fut_step):

      if typ == 0:
        pre = pred[stp, :2 * lm_size]
        posex, posey, visib = (pre[:lm_size], pre[lm_size:],
                               np.ones_like(mask[seen_step + stp, :]))
      else:
        posex, posey, visib = (pose[seen_step + stp, :lm_size],
                               pose[seen_step + stp, lm_size:],
                               mask[seen_step + stp, :])

      sample = visualize_lm(posex, posey, visib, vid_path, vid_imgs, stidx,
                            stp, image_size)

      file_name = ('pred_full_' if typ == 0 else 'gt_') + '%04d.png' % stp
      sm.imsave(save_path_tmp + '/' + file_name, sample)

  cmd1 = 'rm ' + save_path_tmp + '/' + 'gt.gif'
  cmd2 = 'ffmpeg -loglevel panic -f image2 -framerate 7 -i '+save_path_tmp\
       +'/gt_%04d.png '+save_path_tmp+'/gt.gif'
  #system(cmd1);
  system(cmd2)

  cmd1 = 'rm ' + save_path_tmp + '/' + 'pred.gif'
  cmd2 = 'ffmpeg -loglevel panic -f image2 -framerate 7 -i '+save_path_tmp\
       +'/pred_%04d.png '+save_path_tmp+'/pred.gif'
  #system(cmd1);
  system(cmd2)

  cmd1 = 'rm ' + save_path_tmp + '/' + 'pred_full.gif'
  cmd2 = 'ffmpeg -loglevel panic -f image2 -framerate 7 -i '+save_path_tmp\
       +'/pred_full_%04d.png '+save_path_tmp+'/pred_full.gif'
  #system(cmd1);
  system(cmd2)

  print('\t%d done.' % idx)


def save_vid(line, pose_path, save_path, pad):
  res = 0
  try:
    tokens = line.split()[0].split('frames')
    ff = sio.loadmat(tokens[0] + 'labels' + tokens[1] + '.mat')
    pose = np.load(pose_path + tokens[1] + '_img.npy')
    bboxes = ff['bbox']
    posey = ff['y']
    posex = ff['x']
    visib = ff['visibility']
    action = ff['action']
    imgs = sorted([f for f in listdir(line.split()[0]) if f.endswith('.jpg')])
    box = np.zeros((4, ))
    boxes = bboxes.round().astype('int32')
    #  box[0]   = boxes[:,0].min()
    #  box[1]   = boxes[:,1].min()
    #  box[2]   = boxes[:,2].max()
    #  box[3]   = boxes[:,3].max()
    base_name = 'img_%s_%s.png' % (tokens[1].split('/')[-1], action[0])
    pre_pose, cur_pose, new_pose = None, None, None
    for j in range(len(imgs)):
      box = bboxes[j].round().astype('int32')
      img = cv2.imread(line.split()[0] + '/' + imgs[j])[:, :, ::-1]
      y1 = box[1] - pad
      y2 = box[3] + pad
      x1 = box[0] - pad
      x2 = box[2] + pad

      cvisib = visib[j]
      if y1 >= 0:
        cposey = posey[j] - y1
      else:
        cposey = posey[j] - box[1]

      if x1 >= 0:
        cposex = posex[j] - x1
      else:
        cposex = posex[j] - box[0]

      if y1 < 0: y1 = 0
      if x1 < 0: x1 = 0

      patch = img[y1:y2, x1:x2]
      plt.clf()
      plt.imshow(patch)
      plt.axis('off')
      plt.tight_layout()
      #    for k in range(cposey.shape[0]):
      #      if cvisib[k]:
      #        plt.plot(cposex[k],cposey[k], 'o')
      """if pre_pose is None:
          new_pose = pose[j]
        else:
          cur_pose = pose[j]
          new_pose = np.zeros_like(cur_pose)
          match_list = range(pre_pose.shape[0])
          for k in range(cur_pose.shape[0]):
            cur_pnt = cur_pose[k]
            min_dis = float('inf')
            min_idx = -1
            for idx in match_list:
              pre_pnt = pre_pose[idx]
              dis = (cur_pnt[0]-pre_pnt[0])**2+(cur_pnt[1]-pre_pnt[1])**2
              if dis < min_dis:
                min_idx = idx
                min_dis = dis
            new_pose[min_idx] = cur_pnt
            match_list.remove(min_idx)      
          
        pre_pose = new_pose"""
      new_pose = pose[j]

      for k in range(new_pose.shape[0]):
        plt.plot(new_pose[k, 0], new_pose[k, 1], 'o')

      plt.savefig(save_path + base_name + '_%05d.png' % j)

    cmd1 = 'ffmpeg -loglevel panic -f image2 -framerate 14 -i '+save_path\
         +base_name+'_%05d.png '+save_path+'img_%s_%s.gif'%(tokens[1].split('/')[-1], action[0])
    system(cmd1)

    print('\t' + line.split()[0] + ' done.')
    res = 1

  except:
    print('\t' + line.split()[0] + ' done.')
  return res


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


def _convert_to_sstable(seqs, output_file):
  with sstable.Builder(output_file) as builder:
    for i, seq in enumerate(seqs):
      feature = {}
      seq_data = tf.train.BytesList(value=[seq])
      feature["seq/encoded"] = tf.train.Feature(bytes_list=seq_data)
      item = tf.train.Example(features=tf.train.Features(feature=feature))
      key = "{:06d}".format(i)
      builder.Add(key, item.SerializeToString())


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


def getvertices(cx, cy, th):
  theta = 0
  theta += th
  vertices = np.zeros((3, 2), dtype='float32')
  rlen = 20 / np.sqrt(3)
  for ind in range(3):
    vertices[ind, 0] = cx + rlen * math.cos(theta)
    vertices[ind, 1] = cy + rlen * math.sin(theta)
    theta += math.pi * 2 / 3

  return vertices


def drawtriangle(vertices, img_size):
  patch = np.zeros((img_size, img_size, 3))
  cv2.fillPoly(patch, np.int32(np.int32([vertices[:-1, None, :]])),
               (255, 255, 255))
  cv2.circle(patch, (int(vertices[-1, 0]), int(vertices[-1, 1])), 2,
             (255, 0, 0), -1)
  return patch.astype('uint8')


def getavgprob(X, X_is, sigma, mask):
  mask = np.concatenate((mask, mask), axis=-1)
  Z = np.exp(-((X - X_is) / sigma)**2)
  return np.log(((mask * Z).sum(axis=-1) / mask.sum(axis=-1)).mean(axis=0))
