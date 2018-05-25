import cv2
import imageio
import numpy as np
import scipy.io as sio
from os import makedirs
from os.path import exists

def get_tube(posey, posex, img, pad=10):
  box = np.array([0, 0, img.shape[1], img.shape[0]])
  box = np.array([posex.min(), posey.min(), posex.max(), posey.max()])

  x1 = box[0] - pad
  x2 = box[2] + pad
  y1 = box[1] - pad
  y2 = box[3] + pad

  h = y2-y1+1
  w = x2-x1+1
  if h > w:
    left_pad = (h - w ) / 2
    right_pad = (h - w) / 2 + (h - w) % 2
    x1 = x1 - left_pad

    if x1 < 0:
      x1 = 0
    x2 = x2 + right_pad

    if x2 > img.shape[1]:
      x2 = img.shape[1]
  elif w > h:
    up_pad = (w - h) / 2
    down_pad = (w - h) / 2 + (w - h) % 2

    y1 = y1 - up_pad
    if y1 < 0:
      y1 = 0

    y2 = y2 + down_pad
    if y2 > img.shape[0]:
      y2 = img.shape[0]

  if y1 >= 0:
    cposey = posey - y1
  else:
    cposey = posey - box[1]

  if x1 >= 0:
    cposex = posex - x1
  else:
    cposex = posex - box[0]

  if y1 < 0:
    y1 = 0
  if x1 < 0:
    x1 = 0

  box = np.array([x1, y1, x2, y2])
  return cposey.astype('int32'), cposex.astype('int32'), box.astype('int32')


for mode in ['train', 'test']:
  image_size = 128
  data_path = './datasets/Human3.6M/'
  f = open(data_path+mode+'_list.txt','r')
  files = f.readlines()
  f.close()
  
  for f in files:
    vid_path = f.split('\n')[0]
    print(vid_path)
    vid = imageio.get_reader(vid_path,'ffmpeg')
    img = vid.get_data(0)
    anno_path = (vid_path.split('Videos')[0]+'MyPoseFeatures/D2_Positions' +
                 vid_path.split('Videos')[1].split('.mp4')[0]+'.mat')
    pose = sio.loadmat(anno_path)['data']
    pose = np.reshape(pose,[pose.shape[0], 32, 2])
    all_posey = pose[:,:,1]
    all_posex = pose[:,:,0]
    tube_path = (vid_path.split('Videos')[0]+'MyPoseFeatures/D2_Positions' +
                 vid_path.split('Videos')[1].split('.mp4')[0]+'.npz')
    all_posey, all_posex, box = get_tube(all_posey, all_posex, img, pad=20)
    np.savez(tube_path,all_posey=all_posey, all_posex=all_posex, box=box)
  
    save_path = vid_path.split('.mp4')[0]+'/'
    for t in xrange(all_posey.shape[0]):
      if not exists(save_path):
        makedirs(save_path)
      img = cv2.resize(vid.get_data(t)[box[1]:box[3],box[0]:box[2],::-1],
                       (image_size, image_size))
      cv2.imwrite(save_path+'/{0:04}.png'.format(t), img)
      print(save_path+'/{0:04}.png'.format(t)+' saved.')

