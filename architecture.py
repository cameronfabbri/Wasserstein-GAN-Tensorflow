import tensorflow as tf
#import tensorflow.contrib.layers as layers
import tensorflow.contrib.layers as layers
import sys
from tf_ops import *

'''
   Leaky RELU
   https://arxiv.org/pdf/1502.01852.pdf
'''
def lrelu(x, leak=0.2, name='lrelu'):
   return tf.maximum(leak*x, x)

def netG(z, batch_size, num_gpu):
   print 'GENERATOR'
   if num_gpu == 0: gpus=['/cpu:0']
   elif num_gpu == 1: gpus=['/gpu:0']
   elif num_gpu == 2: gpus=['/gpu:0', '/gpu:1']
   elif num_gpu == 3: gpus=['/gpu:0', '/gpu:1', '/gpu:2']
   elif num_gpu == 4: gpus=['/gpu:0', '/gpu:1', '/gpu:2', '/gpu:3']

   for d in gpus:
      with tf.device(d):
         #z = layers.fully_connected(z, 4*4*1024, normalizer_fn=layers.batch_norm, activation_fn=tf.identity, scope='g_z')
         z = layers.fully_connected(z, 4*4*1024, normalizer_fn=layers.batch_norm, activation_fn=tf.identity, scope='g_z')
         z = tf.reshape(z, [batch_size, 4, 4, 1024])
         
         conv1 = layers.convolution2d_transpose(z, 512, 5, stride=2, normalizer_fn=layers.batch_norm, activation_fn=tf.identity, scope='g_conv1')
         #with tf.variable_scope('g_conv1'):
         #   conv1 = tf.image.resize_nearest_neighbor(z, [8,8])
         #   conv1 = conv2d(conv1, 512, stride=1, kernel_size=3)
         #   conv1 = batch_norm(conv1)
         #   conv1 = tf.nn.relu(conv1)

         conv2 = layers.convolution2d_transpose(conv1, 256, 5, stride=2, normalizer_fn=layers.batch_norm, activation_fn=tf.identity, scope='g_conv2')
         #with tf.variable_scope('g_conv2'):
         #   conv2 = tf.image.resize_nearest_neighbor(conv1, [16,16])
         #   conv2 = conv2d(conv2, 256, stride=1, kernel_size=3)
         #   conv2 = batch_norm(conv2)
         conv2 = tf.nn.relu(conv2)

         conv3 = layers.convolution2d_transpose(conv2, 128, 5, stride=2, normalizer_fn=layers.batch_norm, activation_fn=tf.identity, scope='g_conv3')
         #with tf.variable_scope('g_conv3'):
         #   conv3 = tf.image.resize_nearest_neighbor(conv2, [32,32])
         #   conv3 = conv2d(conv3, 128, stride=1, kernel_size=3)
         #   conv3 = batch_norm(conv3)
         conv3 = tf.nn.relu(conv3)

         conv4 = layers.convolution2d_transpose(conv3, 3, 5, stride=2, activation_fn=tf.identity, scope='g_conv4')
         #with tf.variable_scope('g_conv4'):
         #   conv4 = tf.image.resize_nearest_neighbor(conv3, [64,64])
         #   conv4 = conv2d(conv4, 3, stride=1, kernel_size=3)
         conv4 = tf.nn.tanh(conv4)
   
   print 'z:',z
   print 'conv1:',conv1
   print 'conv2:',conv2
   print 'conv3:',conv3
   print 'conv4:',conv4
   print
   print 'END G'
   print
   tf.add_to_collection('vars', conv1)
   tf.add_to_collection('vars', conv2)
   tf.add_to_collection('vars', conv3)
   tf.add_to_collection('vars', conv4)

   return conv4 


'''
   Discriminator network
'''
def netD(input_images, batch_size, num_gpu, reuse=False):
   print 'DISCRIMINATOR'

   if num_gpu == 0: gpus=['/cpu:0']
   elif num_gpu == 1: gpus=['/gpu:0']
   elif num_gpu == 2: gpus=['/gpu:0', '/gpu:1']
   elif num_gpu == 3: gpus=['/gpu:0', '/gpu:1', '/gpu:2']
   elif num_gpu == 4: gpus=['/gpu:0', '/gpu:1', '/gpu:2', '/gpu:3']

   sc = tf.get_variable_scope()
   with tf.variable_scope(sc, reuse=reuse):

      for d in gpus:
         with tf.device(d):
            conv1 = layers.conv2d(input_images, 64, 5, stride=2, activation_fn=None, scope='d_conv1')
            #with tf.variable_scope('d_conv1'):
            #   conv1 = conv2d(input_images, 64, stride=2, kernel_size=5)
            conv1 = lrelu(conv1)

            conv2 = layers.conv2d(conv1, 128, 5, stride=2, normalizer_fn=layers.batch_norm, activation_fn=None, scope='d_conv2')
            #with tf.variable_scope('d_conv2'):
            #   conv2 = conv2d(conv1, 128, stride=2, kernel_size=5)
            #   conv2 = batch_norm(conv2)
            conv2 = lrelu(conv2)
   
            conv3 = layers.conv2d(conv2, 256, 5, stride=2, normalizer_fn=layers.batch_norm, activation_fn=tf.identity, scope='d_conv3')
            #with tf.variable_scope('d_conv3'):
            #   conv3 = conv2d(conv2, 256, stride=2, kernel_size=5)
            #   conv3 = batch_norm(conv3)
            conv3 = lrelu(conv3)

            conv4 = layers.conv2d(conv3, 512, 5, stride=2, normalizer_fn=layers.batch_norm, activation_fn=tf.identity, scope='d_conv4')
            #with tf.variable_scope('d_conv4'):
            #   conv4 = conv2d(conv3, 512, stride=2, kernel_size=5)
            #   conv4 = batch_norm(conv4)
            conv4 = lrelu(conv4)
            
            conv5 = layers.conv2d(conv4, 1, 4, stride=1, activation_fn=tf.identity, scope='d_conv5')
            #with tf.variable_scope('d_conv5'):
            #   conv5 = conv2d(conv4, 1, stride=1, kernel_size=4)
            #   conv5 = batch_norm(conv5)
            conv5 = lrelu(conv5)

      print 'input images:',input_images
      print 'conv1:',conv1
      print 'conv2:',conv2
      print 'conv3:',conv3
      print 'conv4:',conv4
      print 'conv5:',conv5
      print 'END D\n'
      tf.add_to_collection('vars', conv1)
      tf.add_to_collection('vars', conv2)
      tf.add_to_collection('vars', conv3)
      tf.add_to_collection('vars', conv4)
      tf.add_to_collection('vars', conv5)
      return conv5

