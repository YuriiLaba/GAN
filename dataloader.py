import torch
import torch.utils.data as data
import os
import numpy as np

from torchvision import transforms
from PIL import Image

class ImageDataset(data.Dataset):
    def __init__(self, color_images_path, b_and_w_images_path, lower_bound=0, upper_bound=1000, transform=None):
        self.color_images = color_images_path
        self.b_and_w_images = b_and_w_images_path
        self.transform = transform
        self.images_color_lst = []
        self.images_b_and_w_lst = []

        filenames_colour = os.listdir(color_images_path)
        # filenames_bw = os.listdir(b_and_w_images_path)

        upper_bound = min(len(filenames_colour), upper_bound)

        for file_name in filenames_colour[lower_bound: upper_bound]:
            self.images_color_lst.append(color_images_path + "/" + file_name)

        # for file_name in filenames_colour[lower_bound: upper_bound]:
        #     self.images_b_and_w_lst.append(b_and_w_images_path + "/" + file_name)


    def __len__(self):
        return len(self.images_color_lst)

    def __getitem__(self, index):
        image_color = Image.open(self.images_color_lst[index])
        image_b_and_w = Image.open(self.images_color_lst[index]).convert('1')

        if self.transform is not None:
            image_color = self.transform(image_color)
            image_b_and_w = self.transform(image_b_and_w)

        return image_color, image_b_and_w

if __name__ == "__main__":
    batch_size = 100

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    dataset = ImageDataset("/home/yuriy/Cats/Cats_color", "/home/yuriy/Cats/Cats_B&W")
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # print(len(loader.dataset))
    # print(dataset.images_b_and_w)
    print(len(dataset.images_color_lst))
    print(len(dataset.images_b_and_w_lst))
    # print(dataset.images[letter[1]])