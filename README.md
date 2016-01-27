DCGAN in Tensorflow
====================

Tensorflow implementation of [Deep Convolutional Generative Adversarial Networks](http://arxiv.org/abs/1511.06434) which is a stabilize Generative Adversarial Networks. The referenced torch code can be found [here](https://github.com/soumith/dcgan.torch).

![alt tag](DCGAN.png)

*To avoid the fast convergence of D (discriminator) network, G (generatior) network is updatesd twice for each D network update which is a different from original paper.*


Prerequisites
-------------

- Python 2.7 or Python 3.3+
- [Tensorflow](https://www.tensorflow.org/)
- [SciPy](http://www.scipy.org/install.html)


Usage
-----

First, download dataset with:

    $ mkdir data
    $ python download.py --datasets celebA

To train a model with celebA dataset:

    $ python main.py --dataset celebA --is_train True

To test with an existing model:

    $ python main.py --dataset celebA

Or, you can use your own dataset by:

    $ mkdir data/DATASET_NAME
    ... add images to data/DATASET_NAME ...
    $ python main.py --dataset DATASET_NAME --is_train True
    $ python main.py --dataset DATASET_NAME


Results
-------

![result](https://media.giphy.com/media/l3nW2iYprSsXtagYo/giphy.gif)

After 6th epoch:

![result3](assets/result_16_01_04_.png)

(in progress)

Author
------

Taehoon Kim / [@carpedm20](http://carpedm20.github.io/)
