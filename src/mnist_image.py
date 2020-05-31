import os
import pickle
import numpy as np
from PIL import Image
from dataset.mnist import load_mnist

def img_show(img, isReshape=True):
    if isReshape:
        img = np.reshape(img, (28, 28))
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

def get_data(count=0):
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
    if count != 0:
        x_test = x_test[0:count];
        t_test = t_test[0:count];
    return x_test, t_test

def init_network(file_name):
    with open(file_name, 'rb') as f:
        network = pickle.load(f)

    return network

