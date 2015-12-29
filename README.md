End-To-End Memory Networks in Tensorflow
========================================

Tensorflow implementation of [End-To-End Memory Networks](http://arxiv.org/abs/1503.08895v4) for language modeling (see Section 5). The original torch code from Facebook can be found [here](https://github.com/facebook/MemNN/).

![alt tag](http://i.imgur.com/nv89JLc.png)


Prerequisites
-------------

This code requires [Tensorflow](https://www.tensorflow.org/). There is a set of sample Penn Tree Bank (PTB) corpus in `data` directory, which is a popular benchmark for measuring quality of these models. But you can use your own text data set which should be formated like [this](data/).


Usage
-----

To train a model with 6 hops and memory size of 100 (best model described in the paper), run the following command:

    $ python main.py --nhop 6 --mem_size 100

To see all training options, run:

    $ python main.py --help

(Optional) If you want to see a progress bar, install `progress` with `pip`:

    $ pip install progress
    $ python main.py --show True --nhop 6 --mem_size 100


Author
------

Taehoon Kim / [@carpedm20](http://carpedm20.github.io/)
