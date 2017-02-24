'''

Operations used for data management

'''

import tensorflow as tf
import numpy as np
import math

def unnormalizeImage(img, n='tanh'):
   if n == 'tanh':
      img = (img+1.)/2.
      return np.uint8(255.0/img.max())
   if n == 'norm': return np.uint8(255.0*img)

def normalizeImage(img, n='tanh'):
   if n == 'tanh': return img/127.5 - 1. # normalize between -1 and 1
   if n == 'norm': return img/255.0      # normalize between 0 and 1

def saveImage(img, dataset, n='tanh'):
   cv2.imwrite(unnormalizeImage(img, n), path)

