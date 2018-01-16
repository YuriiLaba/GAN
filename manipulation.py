# import matplotlib
# matplotlib.use('Agg')
#
# import matplotlib.pyplot as plt
# from math import ceil
# import torch
# from torch.autograd import Variable
from PIL import Image
import numpy as np
import os


def save_images(images, size=(32, 32), filename='sample', width=8):
    dirs = filename.split("/")
    path = os.getcwd()
    for dir in dirs[1:-1]:
        path += "/"
        path += dir
        os.makedirs(path, exist_ok=True)
    path += "/"
    path += dirs[-1]
    print(path)
    for i, img in enumerate(images):
        img = img.reshape(size).swapaxes(0,2)

        pil_img = Image.fromarray(img)
        # plt.subplot(ceil(len(images) / width), width, i + 1)
        # plt.axis('off')
        # plt.imshow(img, cmap='gray')
        pil_img.save(path + "_" + str(i) + ".png")
    #     plt.subplot(ceil(len(images) / width), width, i + 1)
    #     plt.axis('off')
    #     plt.imshow(img)
    # plt.savefig(filename)


if __name__ == "__main__":

    arr = np.asarray(Image.open("/home/dzvinka/PycharmProjects/GAN/IMG_20171021_113009.jpg"))

    arr = np.expand_dims(arr, axis=0)

    save_images(arr, filename="/img/try/sample")