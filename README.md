# Wasserstein GAN Tensorflow
Implementation of [Wasserstein GAN](https://arxiv.org/abs/1701.07875) in Tensorflow. Official repo for
the paper can be found [here](https://github.com/martinarjovsky/WassersteinGAN)

Requirements
* Python 2.7
* [Tensorflow v1.0](https://www.tensorflow.org/)
* [CelebA dataset](https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip)

#### Training
Training is pretty slow due to the small learning rate and multiple updates of the critic for one
update of the generator. Preloading the data (not reading images from disk every step) helps speed
it up a bit.

I noticed that clipping the weights of the critic to [-0.1, 0.1] like they do in the paper caused the
critic and generator loss to not really change, although image quality was increasing. I found that instead
clipping the weights to [-0.05, 0.05] worked a bit better.

#### Data


#### Tensorboard
Tensorboard logs are stored in `checkpoints/celeba/logs`. I am updating Tensorboard every step as training
isn't completely stable yet. *These can get very big*, around 50GB. See around line 115 in `train.py` to
change how often logs are committed.

#### Results



