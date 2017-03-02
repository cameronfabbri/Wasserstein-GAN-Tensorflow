import scipy.misc as misc
import time
import tensorflow as tf
from architecture import netD, netG
import numpy as np
import random
import ntpath
import sys
import cv2
import os
from skimage import color

import loadceleba

def _read_input(filename_queue):
   class DataRecord(object):
      pass

   reader             = tf.WholeFileReader()
   key, value         = reader.read(filename_queue)
   record             = DataRecord()
   decoded_image      = tf.image.decode_jpeg(value, channels=3)
   decoded_image_4d   = tf.expand_dims(decoded_image, 0)
   resized_image      = tf.image.resize_bilinear(decoded_image_4d, [96, 96])
   record.input_image = tf.squeeze(resized_image, squeeze_dims=[0])
   #cropped_image      = tf.cast(tf.image.crop_to_bounding_box(decoded_image, 55, 35, 64, 64),tf.float32)
   cropped_image      = tf.cast(tf.image.central_crop(decoded_image, 0.6), tf.float32)
   decoded_image_4d   = tf.expand_dims(cropped_image, 0)
   resized_image      = tf.image.resize_bilinear(decoded_image_4d, [64, 64])
   record.input_image = tf.squeeze(resized_image, squeeze_dims=[0])

   return record

def read_input_queue(filename_queue):
   read_input = _read_input(filename_queue)
   num_preprocess_threads = 8
   min_queue_examples = int(0.1 * 100)
   print("Shuffling")
   input_image = tf.train.shuffle_batch([read_input.input_image],
                                        batch_size=64,
                                        num_threads=num_preprocess_threads,
                                        capacity=min_queue_examples + 8 * 64,
                                        min_after_dequeue=min_queue_examples)
   input_image = input_image/127.5 - 1
   return input_image

'''
   Builds the graph and sets up params, then starts training
'''
def buildAndTrain(info):

   checkpoint_dir = info['checkpoint_dir']
   batch_size     = info['batch_size']
   data_dir       = info['data_dir']
   dataset        = info['dataset']
   load           = info['load']
   gray           = info['load']

   # placeholders for data going into the network
   global_step = tf.Variable(0, name='global_step', trainable=False)
   z           = tf.placeholder(tf.float32, shape=(batch_size, 100), name='z')

   train_images_list = loadceleba.load(load=False, data_dir=data_dir)['images']
   filename_queue = tf.train.string_input_producer(train_images_list)
   real_images = read_input_queue(filename_queue)

   # generated images
   gen_images = netG(z, batch_size)

   # get the output from D on the real and fake data
   errD_real = netD(real_images, batch_size)
   errD_fake = netD(gen_images, batch_size, reuse=True) # gotta pass reuse=True to reuse weights

   # cost functions
   errD = tf.reduce_mean(errD_real - errD_fake)
   errG = tf.reduce_mean(errD_fake)

   # tensorboard summaries
   tf.summary.scalar('d_loss', errD)
   tf.summary.scalar('g_loss', errG)
   tf.summary.image('real_images', real_images, max_outputs=batch_size)
   tf.summary.image('generated_images', gen_images, max_outputs=batch_size)
   merged_summary_op = tf.summary.merge_all()

   # get all trainable variables, and split by network G and network D
   t_vars = tf.trainable_variables()
   d_vars = [var for var in t_vars if 'd_' in var.name]
   g_vars = [var for var in t_vars if 'g_' in var.name]

   # clip weights in D
   clip_values = [-0.005, 0.005]
   clip_discriminator_var_op = [var.assign(tf.clip_by_value(var, clip_values[0], clip_values[1])) for
      var in d_vars]

   # optimize G
   G_train_op = tf.train.RMSPropOptimizer(learning_rate=0.00005).minimize(errG, var_list=g_vars, global_step=global_step, colocate_gradients_with_ops=True)

   # optimize D
   D_train_op = tf.train.RMSPropOptimizer(learning_rate=0.00005).minimize(errD, var_list=d_vars, global_step=global_step, colocate_gradients_with_ops=True)

   saver = tf.train.Saver(max_to_keep=1)
   #init  = tf.global_variables_initializer()
   init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
   
   sess  = tf.Session()
   sess.run(init)

   # write out logs for tensorboard to the checkpointSdir
   summary_writer = tf.summary.FileWriter(checkpoint_dir+dataset+'/'+'logs/', graph=tf.get_default_graph())

   tf.add_to_collection('G_train_op', G_train_op)
   tf.add_to_collection('D_train_op', D_train_op)
   
   # restore previous model if there is one
   ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
   if ckpt and ckpt.model_checkpoint_path:
      print "Restoring previous model..."
      try:
         saver.restore(sess, ckpt.model_checkpoint_path)
         print "Model restored"
      except:
         print "Could not restore model"
         pass
   
   ########################################### training portion

   step = sess.run(global_step)
   
   coord = tf.train.Coordinator()
   threads = tf.train.start_queue_runners(sess, coord=coord)

   while True:
      
      start = time.time()

      # get the discriminator properly trained at the start
      if step < 25 or step % 500 == 0:
         n_critic = 100
      else: n_critic = 5

      # train the discriminator for 5 or 25 runs
      for critic_itr in range(n_critic):
         batch_z = np.random.uniform(-1.0, 1.0, size=[batch_size, 100]).astype(np.float32)
         sess.run(D_train_op, feed_dict={z:batch_z})
         sess.run(clip_discriminator_var_op)

      # now train the generator once! use normal distribution, not uniform!!
      batch_z = np.random.normal(-1.0, 1.0, size=[batch_size, 100]).astype(np.float32)
      sess.run(G_train_op, feed_dict={z:batch_z})

      # now get all losses and summary *without* performing a training step - for tensorboard
      D_loss, G_loss, summary = sess.run([errD, errG, merged_summary_op], feed_dict={z:batch_z})
      summary_writer.add_summary(summary, step)

      print 'step:',step,'D loss:',D_loss,'G_loss:',G_loss,'time:',time.time()-start
      step += 1
    
      '''
      image = sess.run(real_images)[0]
      image = (image+1)
      image *= 127.5
      image = np.clip(image, 0, 255).astype(np.uint8)
      image = np.reshape(image, (64, 64, -1))
      misc.imsave('IMAGE.jpg', image)
      exit()
      '''
      
      if step%1000 == 0:
         print 'Saving model...'
         #saver.save(sess, 'my-model', global_step=step)
         saver.save(sess, checkpoint_dir+'checkpoint-'+str(step))
         saver.export_meta_graph(checkpoint_dir+'checkpoint-'+str(step)+'.meta')
         batch_z  = np.random.uniform(-1.0, 1.0, size=[batch_size, 100]).astype(np.float32)
         gen_imgs = sess.run([gen_images], feed_dict={z:batch_z})

         num = 0
         for image in gen_imgs[0]:
            #img = np.asarray(img)
            #img = (img+1.)/2. # these two lines properly scale from [-1, 1] to [0, 255]
            #img *= 255.0/img.max()
            #cv2.imwrite('images/'+dataset+'/'+str(step)+'_'+str(num)+'.png', img)
            image = (image+1)
            image *= 127.5
            image = np.clip(image, 0, 255).astype(np.uint8)
            image = np.reshape(image, (64, 64, -1))
            misc.imsave('images/'+dataset+'/'+str(step)+'_'+str(num)+'.jpg', image)
            num += 1
            if num == 20:
               break
         print 'Done saving'







