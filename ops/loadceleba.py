import cv2
import numpy as np
from random import shuffle
from tqdm import tqdm
import os
import fnmatch
import cPickle as pickle

'''
   Inputs: An image, either 2D or 3D

   Returns: The image cropped to 'size' from the center
'''
def centerCrop(img, size=64):
   height, width, c = img.shape
   #if height is not 64 or width is not 64:
   #   img = misc
   center = (height/2, width/2)
   size   = size/2
   return img[center[0]-size:center[0]+size,center[1]-size:center[1]+size, :]
   
'''
   Inputs: A directory containing images (can have nested dirs inside) and optional extension

   Outputs: A list of image paths
'''
def getPaths(data_dir, ext='jpg'):
   pattern   = '*.'+ext
   image_list = []
   for d, s, fList in os.walk(data_dir):
      for filename in fList:
         if fnmatch.fnmatch(filename, pattern):
            image_list.append(os.path.join(d,filename))
   return image_list


'''
   Loads the celeba data
'''
def load(
         data_dir='/home/fabbric/data/images/celeba/96x96_images/color/',
         normalize_fn='tanh',
         normalize='tanh',
         load=False,
         crop=True,
   ):
   
   # celeba pickle file contains: data['images'] = ['/path/img1.jpg', '/path/img2.jpg', ... ]
   pkl_file    = data_dir+'celeba.pkl'

   # first, check if a pickle file has been made with the image paths
   if os.path.isfile(pkl_file):
      print 'Pickle file found'
      image_paths = pickle.load(open(pkl_file, 'rb'))
      if load is False: return image_paths
   else:
      print 'Getting paths!'
      image_paths = dict()
      image_paths['images'] = getPaths(data_dir)
      pf   = open(pkl_file, 'wb')
      data = pickle.dumps(image_paths)
      pf.write(data)
      pf.close()
      if not load: return image_paths

   num_images = len(image_paths['images'])
   print num_images,'images'
   image_data = np.empty((num_images, 64, 64, 3), dtype=np.float32)

   print 'Loading data...'
   i = 0
   for image in tqdm(image_paths['images']):

      img = centerCrop(cv2.imread(image).astype('float32'), size=64)
      if normalize == 'tanh': img = img/127.5 - 1. # normalize between -1 and 1
      if normalize == 'norm': img = img/255.0      # normalize between 0 and 1

      image_data[i, ...] = img
      
      i += 1
      #if i == 1000: break
   
   return image_data

