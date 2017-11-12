Deep Vision
===========

We're building a deep learning pipeline to tackle problems in machine vision!

## Set-up

### Data

Currently, we are working with the CIFAR-10 dataset. Download the python version from [here](https://www.cs.toronto.edu/~kriz/cifar.html) and add the directory `cifar-10-batches-py` to your project directory.

### Saving images

By default, in order to save space, we are not including the reconstructed
images in the repo. In order to get them, you would run:

```python
data = Data()
data.save()
```

This outputs the images to your IMG_DIR (specified in `config.py`) and saves
a list of image names to labels as LABEL_FILE (also specified in `config.py`).
