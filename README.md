DCGAN in Tensorflow
====================

Tensorflow implementation of [Deep Convolutional Generative Adversarial Networks](http://arxiv.org/abs/1511.06434). The referenced torch code can be found [here](https://github.com/soumith/dcgan.torch).

![alt tag](DCGAN.png)


Prerequisites
-------------

- Python 2.7 or Python 3.3+
- [Tensorflow](https://www.tensorflow.org/)
- [SciPy](http://www.scipy.org/install.html)
- (Optional) [jpeglib](http://mariz.org/blog/2007/01/26/mac-os-x-decoder-jpeg-not-available/) : Only for Mac users


Usage
-----

First, download dataset with:

    $ mkdir data
    $ python download.py --datasets celebA

To train a model with celebA dataset:

    $ python main.py --dataset celebA --is_train True

To test with a existing model:

    $ python main.py --dataset celebA


Results
-------

In progress


Author
------

Taehoon Kim / [@carpedm20](http://carpedm20.github.io/)
