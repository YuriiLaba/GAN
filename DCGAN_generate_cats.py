from PIL import Image
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam
from torchvision import transforms
import torchvision.datasets as dsets
import torch.nn.functional as F
import numpy as np

from manipulation import save_images

from dataloader import *




class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(3, 32, 3, stride=2, padding=2)
        self.conv_2 = nn.Conv2d(32, 64, 3, stride=2, padding=2)
        self.conv_3 = nn.Conv2d(64, 128, 3, stride=2, padding=2)
        self.conv_4 = nn.Conv2d(128, 64, 3, stride=2, padding=2)
        self.conv_5= nn.Conv2d(64, 1, 3, stride=2, padding=2)
        self.conv_1_bn = nn.BatchNorm2d(32)
        self.conv_2_bn = nn.BatchNorm2d(64)
        self.conv_3_bn = nn.BatchNorm2d(128)
        self.conv_4_bn = nn.BatchNorm2d(64)

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

        self.conv_1 = nn.Conv2d(1, 32, 4, stride=2)
        self.conv_2 = nn.Conv2d(32, 64, 3, stride=2)
        self.conv_3 = nn.Conv2d(64, 128, 3, stride=2)
        self.deconv_1 = nn.ConvTranspose2d(128, 64, 3, stride=2)
        self.deconv_2 = nn.ConvTranspose2d(64, 32, 3, stride=2)
        self.deconv_3 = nn.ConvTranspose2d(32, 1, 4, stride=2)

        self.conv_1_bn = nn.BatchNorm2d(32)
        self.conv_2_bn = nn.BatchNorm2d(64)
        self.conv_3_bn = nn.BatchNorm2d(128)
        self.deconv_1_bn = nn.BatchNorm2d(64)
        self.deconv_2_bn = nn.BatchNorm2d(32)

    def forward(self, x):
        x = F.leaky_relu(self.conv_1(x))
        x = F.leaky_relu(self.conv_2_bn(self.conv_2(x)))
        x = F.leaky_relu(self.conv_3_bn(self.conv_3(x)))
        x = F.leaky_relu(self.deconv_1_bn(self.deconv_1(x)))
        x = F.leaky_relu(self.deconv_2_bn(self.deconv_2(x)))
        x = F.tanh(self.deconv_3(x))
        return x


def train_GAN(use_cuda=False):
    path = "/data/" if use_cuda else "/home/dobosevych/Documents/Cats/"
    train_loader = load_data(path, upper_bound=21000)


    lr = 0.0002
    betas = (0.5, 0.999)
    discriminator = Discriminator()
    generator = Generator()



    if use_cuda:
        discriminator = discriminator.cuda()
        generator = generator.cuda()

    d_optimizer = Adam(discriminator.parameters(), lr=lr, betas=betas)
    g_optimizer = Adam(generator.parameters(), lr=lr, betas=betas)
    criterion = nn.BCELoss()

    num_epochs = 100
    num_of_samples = 100

    for epoch in range(num_epochs):
        for i, (color_images, b_and_w_images) in enumerate(train_loader):
            true_images = Variable(b_and_w_images.view(-1, DIM * DIM))  # discriminator input is 28 * 28
            true_labels = Variable(torch.ones(true_images.size(0)))

            # Sample data from generator
            noise = Variable(
                torch.randn(true_images.size(0), 100))  # generator input is 100  ToDo: HOW DO WE GET 1024????
            fake_labels = Variable(torch.zeros(true_images.size(0)))

            if use_cuda:
                true_images, noise, fake_labels, true_labels = true_images.cuda(), noise.cuda(), fake_labels.cuda(), true_labels.cuda()

            fake_images = generator(noise)

            # Discriminator training
            discriminator.zero_grad()
            out1 = discriminator(true_images)
            true_loss = criterion(out1, true_labels)

            out2 = discriminator(fake_images)
            fake_loss = criterion(out2, fake_labels)

            d_loss = true_loss + fake_loss
            d_loss.backward()
            d_optimizer.step()

            # Sample from generator
            noise = Variable(torch.randn(b_and_w_images.size(0), 100))
            if use_cuda:
                noise = noise.cuda()

            fake_images = generator(noise)
            out = discriminator(fake_images)

            # Generator training
            generator.zero_grad()
            g_loss = criterion(out, true_labels)
            g_loss.backward()
            g_optimizer.step()

            print("Epoch: [{}/{}], Step: [{}/{}]".format(epoch + 1, num_epochs, i + 1, len(train_loader)))

        test_noise = Variable(torch.randn(num_of_samples, 100))

        if use_cuda:
            test_noise = Variable(torch.randn(num_of_samples, 100)).cuda()

        test_images = generator(test_noise)
        filename = "/output/epoch_{}/sample" if use_cuda else "samples/epoch_{}/sample"
        save_images(test_images, filename=filename.format(epoch + 1), width=10, size=(3, 128, 128))



if __name__ == "__main__":
    train_GAN(True)
