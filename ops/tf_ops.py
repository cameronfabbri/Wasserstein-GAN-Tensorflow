'''

Operations commonly used in tensorflow

'''

import tensorflow as tf
import numpy as np
import math


def normalizeImage(img, n='tanh'):
   if n == 'tanh': return img/127.5 - 1. # normalize between -1 and 1
   if n == 'norm': return img/255.0      # normalize between 0 and 1

def saveImage(img, path, dataset):
   img = (img+1.)/2.
   img = 255.0/img.max()
   cv2.imwrite(img, path)

'''
   Leaky RELU
   https://arxiv.org/pdf/1502.01852.pdf
'''
def lrelu(x, leak=0.2, name='lrelu'):
   return tf.maximum(leak*x, x)

'''
   Regular relu
'''
def relu(x, name='relu'):
   return tf.nn.relu(x, name)

'''
   Tanh
'''
def tanh(x, name='tanh'):
   return tf.nn.tanh(x, name)

'''
   Sigmoid
'''
def sig(x, name='sig'):
   return tf.nn.sigmoid(x, name)

'''
   Places a variable on the GPU
'''
def _variable_on_gpu(name, shape, initializer):
   with tf.device('/gpu:0'):
      var = tf.get_variable(name, shape, initializer=initializer)
   return var

'''
   Creates a variable with weight decay
'''
def _variable_with_weight_decay(name, shape, stddev, wd):
   var = _variable_on_gpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))
   if wd:
      weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
      weight_decay.set_shape([])
      tf.add_to_collection('losses', weight_decay)
   return var
