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
        pil_img.save(path + "_" + str(i) + ".png")