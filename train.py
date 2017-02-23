import tensorflow as tf
from architecture import netD, netG
import numpy as np
import random
import ntpath
import sys
import cv2
import os

sys.path.insert(0, 'config/')
sys.path.insert(0, 'ops/')
import celeba
import mnist
import pokemon

'''
   Builds the graph and sets up params, then starts training
'''
def buildAndTrain(info):
   
   global_step = tf.Variable(0, name='global_step', trainable=False)

   checkpoint_dir = info['checkpoint_dir']
   batch_size     = info['batch_size']
   dataset        = info['dataset']
   load           = info['load']
   gray           = info['load']

   # load data
   if dataset == 'mnist':
      image_data = mnist.load(split='all')
      real_images = tf.placeholder(tf.float32, shape=(batch_size, 28, 28, 1), name='real_images')

   elif dataset == 'celeba':
      image_data = loadceleba.load(load=load)
      real_images = tf.placeholder(tf.float32, shape=(batch_size, 64, 64, 3), name='real_images')

   elif dataset == 'pokemon':
      image_data = pokemon.load(load=load)
      real_images = tf.placeholder(tf.float32, shape=(batch_size, 64, 64, 3), name='real_images')

   z = tf.placeholder(tf.float32, shape=(batch_size, 100), name='z')

   # generated images
   gen_images = netG(z, dataset, batch_size)

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
   #clip_values = [-0.01, 0.01]
   clip_values = [-0.005, 0.005]
   clip_discriminator_var_op = [var.assign(tf.clip_by_value(var, clip_values[0], clip_values[1])) for
      var in d_vars]

   # optimize G
   G_train_op = tf.train.RMSPropOptimizer(learning_rate=0.00005).minimize(errG, var_list=g_vars, global_step=global_step)

   # optimize D
   D_train_op = tf.train.RMSPropOptimizer(learning_rate=0.00005).minimize(errD, var_list=d_vars, global_step=global_step)

   # change to use a fraction of memory
   #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
   init      = tf.global_variables_initializer()
   #sess      = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
   sess = tf.Session()
   sess.run(init)

   # write out logs for tensorboard to the checkpointSdir
   summary_writer = tf.summary.FileWriter(checkpoint_dir+dataset+'/'+'logs/', graph=tf.get_default_graph())

   # only keep one model
   saver = tf.train.Saver(max_to_keep=1)
   ckpt = tf.train.get_checkpoint_state(checkpoint_dir+dataset+'/')

   # restore previous model if there is one
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
   
   while True:

      # get the discriminator properly trained at the start
      if step < 25 or step % 500 == 0:
         n_critic = 100
      else: n_critic = 5

      # train the discriminator for 5 or 25 runs
      for critic_itr in range(n_critic):
         batch_real_images = random.sample(image_data, batch_size)
         batch_z = np.random.uniform(-1.0, 1.0, size=[batch_size, 100]).astype(np.float32)
         sess.run(D_train_op, feed_dict={real_images:batch_real_images, z:batch_z})
         sess.run(clip_discriminator_var_op)

      # now train the generator once! use normal distribution, not uniform!!
      batch_z = np.random.normal(-1.0, 1.0, size=[batch_size, 100]).astype(np.float32)
      sess.run(G_train_op, feed_dict={z:batch_z})

      # now get all losses and summary *without* performing a training step - for tensorboard
      D_loss, G_loss, summary = sess.run([errD, errG, merged_summary_op],
                                          feed_dict={real_images:batch_real_images, z:batch_z})

      summary_writer.add_summary(summary, step)

      print 'step:',step,'D loss:',D_loss,'G_loss:',G_loss

      step += 1

      if step%1000 == 0:
         print 'Saving model...'
         saver.save(sess, checkpoint_dir+dataset+'/checkpoint-'+str(step), global_step=global_step)
         
         batch_z  = np.random.uniform(-1.0, 1.0, size=[batch_size, 100]).astype(np.float32)
         gen_imgs = sess.run([gen_images], feed_dict={z:batch_z})

         num = 0
         for img in gen_imgs[0]:
            # TODO make this cleaner, call a save function with the dataset name and image and step etc
            img = np.asarray(img)
            # JUST FOR MNIST
            #img *= 1.0/img.max()
            img = (img+1.)/2. # these two lines properly scale from [-1, 1] to [0, 255]
            img *= 255.0/img.max()
            cv2.imwrite('images/'+dataset+'/'+str(step)+'_'+str(num)+'.png', img)
            num += 1
            if num == 20:
               break
         print 'Done saving'







