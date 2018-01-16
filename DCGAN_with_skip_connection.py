#COMMON GAN WITH FULY CONNECTED LAYERS USED FOR IMAGE GENERATION

from PIL import Image
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam
from torchvision import transforms
import torchvision.datasets as dsets
import torch.nn.functional as F
import numpy as np
from vgg import Vgg16
from manipulation import save_images
import torchvision.models as models

from dataloader import *

def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

def load_data(path, lower_bound=0, upper_bound=1000):
    batch_size = 100

    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    dataset = ImageDataset(path + "Cats_color_128", path + "Cats_B&W_128", transform=transform, lower_bound=lower_bound, upper_bound=upper_bound)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader



class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(3, 32, 3, stride=2, padding=2)
        self.conv_2 = nn.Conv2d(32, 64, 3, stride=2, padding=2)
        self.conv_3 = nn.Conv2d(64, 128, 3, stride=2, padding=2)
        self.conv_4 = nn.Conv2d(128, 64, 3, stride=2, padding=2)
        self.conv_5 = nn.Conv2d(64, 1, 3, stride=2, padding=2)
        self.conv_1_bn = nn.BatchNorm2d(32)
        self.conv_2_bn = nn.BatchNorm2d(64)
        self.conv_3_bn = nn.BatchNorm2d(128)
        self.conv_4_bn = nn.BatchNorm2d(64)
        self.fc = nn.Linear(1000, 1)

        # should be corrected if new image arrive
        # self.fc = nn.Linear(dim * 4 * 4, 1)

    def forward(self, x):
        x = F.leaky_relu(self.conv_1(x), 0.1)
        x = F.leaky_relu(self.conv_2_bn(self.conv_2(x)), 0.1)
        x = F.leaky_relu(self.conv_3_bn(self.conv_3(x)), 0.1)
        x = F.leaky_relu(self.conv_4_bn(self.conv_4(x)), 0.1)
        x = self.conv_5(x)
        x = x.view(x.size(0), -1).mean(1)
        x = F.sigmoid(x)

        return x

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_1 = nn.Conv2d(1, 32, 4, stride=2, bias=False)
        self.conv_2 = nn.Conv2d(32, 64, 3, stride=2, bias=False)
        self.conv_3 = nn.Conv2d(64, 128, 3, stride=2, bias=False)
        self.conv_4 = nn.Conv2d(128, 128, 3, stride=2, bias=False)
        self.deconv_1 = nn.ConvTranspose2d(128, 128, 3, stride=2, bias=False)
        self.deconv_2 = nn.ConvTranspose2d(128, 64, 3, stride=2, bias=False)
        self.deconv_3 = nn.ConvTranspose2d(64, 32, 3, stride=2, bias=False)
        self.deconv_4 = nn.ConvTranspose2d(32, 3, 4, stride=2, bias=False)

        self.conv_1_bn = nn.BatchNorm2d(32)
        self.conv_2_bn = nn.BatchNorm2d(64)
        self.conv_3_bn = nn.BatchNorm2d(128)
        self.conv_4_bn = nn.BatchNorm2d(128)
        self.deconv_1_bn = nn.BatchNorm2d(128)
        self.deconv_2_bn = nn.BatchNorm2d(64)
        self.deconv_3_bn = nn.BatchNorm2d(32)

    def forward(self, x):
        # print('x', x.size())
        x1 = F.relu(self.conv_1(x))
        # print('x1', x1.size())
        x2 = F.relu(self.conv_2_bn(self.conv_2(x1)))
        # print('x2', x2.size())
        x3 = F.relu(self.conv_3_bn(self.conv_3(x2)))
        # print('x3', x3.size())
        x4 = F.relu(self.conv_4_bn(self.conv_4(x3)))
        # print('x4', x4.size())
        x5 = F.relu(self.deconv_1_bn(self.deconv_1(x4))) + x3
        # print('x5', x5.size())
        x6 = F.relu(self.deconv_2_bn(self.deconv_2(x5))) + x2
        # print('x6', x6.size())
        x7 = F.relu(self.deconv_3_bn(self.deconv_3(x6))) + x1
        # print('x7', x7.size())
        x8 = F.tanh(self.deconv_4(x7))
        # print('x8', x8.size())
        return x8


def train_GAN(use_cuda=False):
    path = "/data/" if use_cuda else "/home/dobosevych/Documents/Cats/"
    train_loader = load_data(path, upper_bound=21000)
    test_loader = load_data(path, lower_bound=21000, upper_bound=22000)

    lr = 0.0002
    betas = (0.5, 0.999)
    discriminator = Discriminator()
    generator = Generator()


    test_images_color, test_images_bw = next(iter(test_loader))
    test_images_bw = Variable(test_images_bw)

    if use_cuda:
        test_images_bw = test_images_bw.cuda()

    if use_cuda:
        discriminator = discriminator.cuda()
        generator = generator.cuda()

    d_optimizer = Adam(discriminator.parameters(), lr=lr, betas=betas)
    g_optimizer = Adam(generator.parameters(), lr=lr, betas=betas)
    criterion_BCE = nn.BCELoss()
    criterion_L1 = nn.L1Loss()

    num_epochs = 100
    num_of_samples = 100

    for epoch in range(num_epochs):
        for i, (color_images, b_and_w_images) in enumerate(train_loader):
            minibatch = color_images.size(0)
            color_images = Variable(color_images)

            # color_images_features = F.batch_norm(color_images)
            # color_images_features = vgg(color_images)

            b_and_w_images = Variable(b_and_w_images)
            labels_1 = Variable(torch.ones(minibatch))
            labels_0 = Variable(torch.zeros(minibatch))

            if use_cuda:
                color_images, b_and_w_images, labels_0, labels_1 = color_images.cuda(), b_and_w_images.cuda(), labels_0.cuda(), labels_1.cuda()#, damaged.cuda()

            # Generator training
            generated_images = generator(b_and_w_images)
            out = discriminator(generated_images)


            loss_img = criterion_L1(generated_images, color_images)
            loss_1 = criterion_BCE(out, labels_1)
            g_loss = loss_1 + loss_img
            # g_loss = styleloss + loss_1
            g_loss.backward()
            g_optimizer.step()

            # Discriminator training
            generated_images = generator(b_and_w_images)
            discriminator.zero_grad()
            out_0 = discriminator(generated_images)
            loss_0 = criterion_BCE(out_0, labels_0)

            out_1 = discriminator(color_images)
            loss_1 = criterion_BCE(out_1, labels_1)

            d_loss = loss_0 + loss_1
            d_loss.backward()
            d_optimizer.step()

            print("Epoch: [{}/{}], Step: [{}/{}]".format(epoch + 1, num_epochs, i + 1, len(train_loader)))


        test_images_colored = generator(test_images_bw)
        test_images_colored = test_images_colored.view(num_of_samples, 3, 128, 128)
        filename_colored = "/output/epoch_{}/colored/" if use_cuda else "samples/epoch_{}/colored/"
        filename_bw = "/output/epoch_{}/black_and_white/" if use_cuda else "samples/epoch_{}/black_and_white/"
        filename_color = "/output/epoch_{}/incolor/" if use_cuda else "samples/epoch_{}/incolor/"

        save_images(test_images_colored.data.cpu().numpy(), path=filename_colored.format(epoch + 1))
        save_images(test_images_bw.data.cpu().numpy(), path=filename_bw.format(epoch + 1))
        save_images(test_images_color.numpy(), path=filename_color.format(epoch + 1))


if __name__ == "__main__":
    train_GAN(True)