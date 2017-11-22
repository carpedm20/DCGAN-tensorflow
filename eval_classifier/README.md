# MNIST Classifier for Evaluation

Accuracy of the MNIST Classifier after training ~ 99.2%

Steps to evaluate your image:
* Generate sample image from your DCGAN model
* Train the Eval Classifier
* Run the generated image on the Eval Classifier

## Generate sample image from your DCGAN model
The changes to DCGAN for this part is only on `main.py`.

To generate test images (after you train your model, from the DCGAN root directory):

`python main.py --dataset mnist --input_heigh=28 --output_height=28 --generate`

This will generate 2x64 images (Since 64 is the number of batches the DCGAN model accept).
You can change this by changing the `n_samples` in `main.py`

The generated image will be located in `eval_classifier\testdata`

## Train the Eval Classifier
In the `eval_classifier` directory:

`python main.py --train`

## Run the generated image on the Eval Classifier
`python main.py`

It'll print out the accuracy.

## Current Result

* Original DCGAN: 98,44%
* Dense DCGAN: 88,28%
* DCGAN + 1 Extra Dense layer:  99,22%
