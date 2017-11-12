"""
data.py: Create training /test data from raw CIFAR-10 batches.
"""

import pickle
import glob
import os
from config import constants

import numpy as np
from skimage.io import imsave

class Data:
    def __init__(self):
        train, test = self.load_data()
        self.X_train = train[0]
        self.y_train = train[1]
        self.X_test = test[0]
        self.y_test = test[1]

    def load_data(self, dir=constants.CIFAR, one_hot=False):

        # load train data
        print(f"Loading train batches...")
        for i in range(1, 6):
            path = os.path.join(constants.CIFAR, f'data_batch_{i}')
            data, labels = self.load_batch(path)

            if i == 1:
                X_train = data
                y_train = labels

            else:
                X_train = np.concatenate([X_train, data], axis=0)
                y_train = np.concatenate([y_train, labels], axis=0)

        # load test data
        print(f"Loading test batches...")
        path = os.path.join(constants.CIFAR, 'test_batch')
        X_test, y_test = self.load_batch(path)

        # RGB
        img_R = X_train[:, :1024]
        img_G = X_train[:, 1024:2048]
        img_B = X_train[:, 2048:]
        X_train = np.dstack((img_R, img_G, img_B)) / 255.
        X_train = np.reshape(X_train, [-1, 32, 32, 3])

        img_R = X_test[:, :1024]
        img_G = X_test[:, 1024:2048]
        img_B = X_test[:, 2048:]
        X_test = np.dstack((img_R, img_G, img_B)) / 255.
        X_test = np.reshape(X_test, [-1, 32, 32, 3])

        if one_hot:
            y_train = to_categorical(y_train, 10)
            y_test = to_categorical(y_test, 10)

        return (X_train, y_train), (X_test, y_test)

    @staticmethod
    def load_batch(fname):
        """Unpacks a CIFAR-10 file."""

        with open(fname, "rb") as f:
            result = pickle.load(f, encoding='latin-1')

        data = result['data']
        labels = result['labels']
        return data, labels

    def save_as_image(img_flat, fname):
        """Saves a data blob as an image file."""

        # consecutive 1024 entries store color channels of 32x32 image
        img_R = img_flat[0:1024].reshape((32, 32))
        img_G = img_flat[1024:2048].reshape((32, 32))
        img_B = img_flat[2048:3072].reshape((32, 32))
        img = np.dstack((img_R, img_G, img_B))

        imsave(os.path.join(constants.IMG_DIR, fname), img)

    def save():
        """Saves all images and labels."""

        labels = {}

        # load train data
        train_wildcard = os.path.join(constants.CIFAR, "*_batch*")
        for fname in glob.glob(train_wildcard):
            data, labels = unpack_file(fname)

            for i in range(10000):
                img_flat = data["data"][i]
                fname = data["filenames"][i]
                label = data["labels"][i]

                # save the image and store the label
                save_as_image(img_flat, fname)
                labels[fname] = label

        # write out labels file
        with open(constants.LABEL_FILE, "w") as f:
            for (fname, label) in labels.items():
                f.write("{0} {1}\n".format(fname, label))