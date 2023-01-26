# Flower Classification

This project is an attempt to classify flowers using a neural network.
Dataset is from https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html?ref=hackernoon.com. Nilsback, M-E. and Zisserman, A.
Model requires the dataset images https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz.
Model also requires the image labels from https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat.

Having the ability to identify plant species without the help of an expert is valuable. A model would allow unskilled persons to contribute to the documenting of species in an area far wider than a few experts alone could cover. Monitoring of the spread of species is important in quantifying the effects of climate change and in preserving the more threatened species.

This model attempts to classify pictures of 102 different classes of flowers.

Database was separated into classes by running separate.py, which uses the imagelabels.mat to classify each image.
separate.py also splits the images into train/val/test folders.

## Model is contained in git repo as 'model.tflite' however it can be remade using the below commands.

## Training model
* Install requirements in requirements.txt.
* Run train.py.
* Run convert.py

## Hosting locally
* Run predict.py
* (I ran out of time)
