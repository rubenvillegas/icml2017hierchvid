import cv2
import scipy.io as sio
import numpy as np
from os import listdir

for mode in ['train', 'test']:
  vid_path = './datasets/PennAction/frames/'
  ann_path = './datasets/PennAction/labels/'
  pad = 5
  f = open('./datasets/PennAction/'+mode+'_list.txt','r')
  lines = f.readlines()
  f.close()
  numvids=len(lines)

  for i, line in enumerate(lines):
    tokens = line.split()[0].split('frames')
    ff = sio.loadmat(tokens[0]+'labels'+tokens[1]+'.mat')
    bboxes = ff['bbox']
    posey = ff['y']
    posex = ff['x']
    visib = ff['visibility']
    imgs = sorted([f for f in listdir(line.split()[0]) if f.endswith('.jpg')])
    box = np.zeros((4,), dtype='int32')
    bboxes = bboxes.round().astype('int32')
  
    if len(imgs) > bboxes.shape[0]:
      bboxes = np.concatenate((bboxes,bboxes[-1][None]),axis=0)
  
    box[0] = bboxes[:,0].min()
    box[1] = bboxes[:,1].min()
    box[2] = bboxes[:,2].max()
    box[3] = bboxes[:,3].max()
  
    for j in xrange(len(imgs)):
      img = cv2.imread(line.split()[0]+'/'+imgs[j])
      y1 = box[1] - pad
      y2 = box[3] + pad
      x1 = box[0] - pad
      x2 = box[2] + pad
  
      h = y2 - y1 + 1
      w = x2 - x1 + 1
      if h > w:
        left_pad  = (h - w) / 2
        right_pad = (h - w) / 2 + (h - w)%2
  
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
  
      cvisib = visib[j]
      if y1 >= 0:
        cposey = posey[j] - y1
      else:
        cposey = posey[j] - box[1]
  
      if x1 >= 0:
        cposex = posex[j] - x1
      else:
        cposex = posex[j] - box[0]
  
      if y1 < 0:
        y1 = 0
      if x1 < 0:
        x1 = 0
  
      patch = img[y1:y2,x1:x2]
      bboxes[j] = np.array([x1, y1, x2, y2])
      posey[j] = cposey
      posex[j] = cposex
      cv2.imwrite(line.split()[0]+'/'+imgs[j].split('.')[0]+'_cropped.png', patch)
  
    ff['bbox'] = bboxes
    ff['y'] = posey
    ff['x'] = posex
    np.savez(ann_path+line.split('/')[-1].split()[0]+'.npz', **ff)
    print(str(i)+'/'+str(numvids)+' '+mode+' processed')
  
