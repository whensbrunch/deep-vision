import os
from collections import namedtuple

_curr_dir = os.path.dirname(os.path.realpath(__file__))
_constants = {
    'CIFAR': os.path.join(_curr_dir, 'cifar-10-batches-py'),
    'IMG_DIR': os.path.join(_curr_dir, 'images'),
    'LABEL_FILE': os.path.join(_curr_dir, 'labels.txt')
}
constants = (namedtuple('Constants', _constants)(**_constants))