import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from math import ceil
import torch
from torch.autograd import Variable


def save_images(images, size=(32, 32), filename='sample.png', width=8):
    for i, img in enumerate(images):
        img = img.reshape(size)
        plt.subplot(ceil(len(images) / width), width, i + 1)
        plt.axis('off')
        plt.imshow(img, cmap='gray')
    plt.savefig(filename)