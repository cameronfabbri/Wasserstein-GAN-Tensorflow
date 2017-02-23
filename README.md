# Wasserstein GAN Tensorflow
Implementation of [Wasserstein GAN](https://arxiv.org/abs/1701.07875) in Tensorflow. Official repo for
the paper can be found [here](https://github.com/martinarjovsky/WassersteinGAN).

#### Outline
* Results
* Training
* Data
* Tensorboard

___

Requirements
* Python 2.7
* [Tensorflow v1.0](https://www.tensorflow.org/)
* [CelebA dataset](https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip)

#### Results
Here are some non cherry-picked generated images after ~120,000 iterations. Images started to get a tad
blurry after ~100,000 iterations. The loss in the graphs shows the critic was starting to get worse,
but both were generally converging.

![img](http://i.imgur.com/AApFex3.jpg)

Critic loss

![d](http://i.imgur.com/Mtx7rlK.png)

Generator loss

![g](http://i.imgur.com/bJBQhBX.png)

#### Training
Training is pretty slow due to the small learning rate and multiple updates of the critic for one
update of the generator. Preloading the data (not reading images from disk every step) helps speed
it up a bit. These were trained on a GTX-1080 for about 24 hours.

I noticed that clipping the weights of the critic to [-0.1, 0.1] like they do in the paper caused the
critic and generator loss to not really change, although image quality was increasing. I found that instead
clipping the weights to [-0.05, 0.05] worked a bit better, showing better image quality and convergence.

#### Data
Standard practice is to resize the CelebA images to 96x96 and the crop a center 64x64 image. `loadceleba.py`
takes as input the directory to your images, and will resize them upon loading. To load the entire dataset
at the start instead of reading from disk each step, you will need about 200000\*64\*64\*3\*3 bytes = ~7.5
GB of RAM.

#### Tensorboard
Tensorboard logs are stored in `checkpoints/celeba/logs`. I am updating Tensorboard every step as training
isn't completely stable yet. *These can get very big*, around 50GB. See around line 115 in `train.py` to
change how often logs are committed.



