import cv2
import pandas as pd
import cPickle as pickle
import pprint
import numpy as np

def load(mnist_file='/home/fabbric/data/images/mnist/mnist.pkl',
         split='all'):

   mnist = pd.read_pickle(mnist_file)

   # train, val, test
   df = pd.DataFrame(list(mnist), columns=['images', 'labels'])

   images = df['images']
   labels = df['labels']
   
   train_images_ = images[0]
   val_images_   = images[1]
   test_images_  = images[2]

   train_labels_ = labels[0]
   val_labels_   = labels[1]
   test_labels_  = labels[2]

   train_images = np.empty((len(train_images_), 28, 28, 1), dtype=np.float32)
   test_images  = np.empty((len(test_images_), 28, 28, 1), dtype=np.float32)
   val_images   = np.empty((len(val_images_), 28, 28, 1), dtype=np.float32)

   # reshape images
   i = 0
   for tr in train_images_:
      tr = np.expand_dims(np.reshape(tr, (28, 28)), 2)
      train_images[i, ...] = tr
      i += 1
   
   i = 0
   for te in test_images_:   
      te = np.expand_dims(np.reshape(te, (28, 28)), 2)
      test_images[i, ...] = tr
      i += 1
      
   i = 0
   for va in val_images_:
      va = np.expand_dims(np.reshape(va, (28, 28)), 2)
      val_images[i, ...] = va
      i += 1

   if split == 'train': return train_images
   if split == 'test':  return test_images
   if split == 'val':   return val_images
   if split == 'all':
      all_images = np.concatenate((train_images, test_images), 0)
      all_images = np.concatenate((all_images, val_images), 0)
      return all_images

def get_train_images(self):
   return train_images

def get_val_images(self):
   return val_images

def get_test_images(self):
   return test_images

def get_train_labels(self):
   return train_labels

def get_val_labels(self):
   return val_labels

def get_test_labels(self):
   return test_labels


