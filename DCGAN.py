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

DIM = 128

def load_data(lower_bound=0, upper_bound=1000):
    batch_size = 100

    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    dataset = ImageDataset("/data/Cats_color_32", "/data/Cats_B&W_32", transform=transform, lower_bound=lower_bound, upper_bound=upper_bound)
    # dataset = ImageDataset("/home/yuriy/Cats/Cats_32/Cats_color_32", "/home/yuriy/Cats/Cats_32/Cats_B&W_32", transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader



class Discriminator(nn.Module):
    def __init__(self, dim, channels=1):
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
        # should be corrected if new image arrive
        x = min(x.view(-1))
        # x = F.sigmoid(self.fc(x))
        x = F.sigmoid((self.conv_5(x)))
        return x

class Generator(nn.Module):
    def __init__(self, dim, channels=1):
        super().__init__()
        self.dim = dim
        # self.fc = nn.Linear(14 * 14, 4 * 4 * dim)
        self.conv_1 = nn.Conv2d(1, 32, 3, stride=2, padding=2)
        self.conv_2 = nn.Conv2d(32, 64, 3, stride=2, padding=2)
        self.conv_3 = nn.Conv2d(64, 128, 3, stride=2, padding=2)
        self.conv_4 = nn.Conv2d(128, 128, 3, stride=2, padding=2)

        self.deconv_1 = nn.ConvTranspose2d(128, 128, 3, stride=2, padding=2)
        self.deconv_2 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=2)
        self.deconv_3 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=2)
        self.deconv_4 = nn.ConvTranspose2d(32, 3, 3, stride=2, padding=2)

        self.conv_1_bn = nn.BatchNorm2d(32)
        self.conv_2_bn = nn.BatchNorm2d(64)
        self.conv_3_bn = nn.BatchNorm2d(128)


    def forward(self, x):
        # x = self.fc(x)
        x = x.view(-1, self.dim, 4, 4)
        x = F.leaky_relu(self.conv_1_bn(self.conv_1(x)))
        x = F.leaky_relu(self.conv_2_bn(self.conv_2(x)))
        x = F.leaky_relu(self.conv_3_bn(self.conv_3(x)))
        x = F.leaky_relu(self.conv_3_bn(self.conv_4(x)))

        x = F.leaky_relu(self.conv_1_bn(self.deconv_1(x)))
        x = F.leaky_relu(self.conv_2_bn(self.deconv_2(x)))
        x = F.leaky_relu(self.conv_3_bn(self.deconv_3(x)))


        x = F.tanh(self.deconv_4(x))

        return x

# # TODO: Correct later
# def make_damaged(images):
#     damaged = images.view(-1, DIM, 28).numpy()
#     new_damaged = np.zeros((images.size(0), 14, 14))
#     for j, img in enumerate(damaged):
#         new_img = Image.fromarray(np.uint8(img))
#         new_img = new_img.resize((14, 14))
#         new_img = np.asarray(new_img)
#         new_damaged[j] = (new_img / 255).astype(np.float)
#     damaged = torch.from_numpy(new_damaged).view(-1, 14 * 14).float()
#     return damaged

def train_GAN(use_cuda=False):

    train_loader = load_data(upper_bound=18000)
    test_loader = load_data(lower_bound=18000, upper_bound=22000)

    lr = 0.0002
    betas = (0.5, 0.999)
    discriminator = Discriminator(DIM)
    generator = Generator(DIM)

    if use_cuda:
        discriminator = discriminator.cuda()
        generator = generator.cuda()

    d_optimizer = Adam(discriminator.parameters(), lr=lr, betas=betas)
    g_optimizer = Adam(generator.parameters(), lr=lr, betas=betas)
    criterion_BCE = nn.BCELoss()
    criterion_MSE = nn.MSELoss()

    num_epochs = 20
    num_of_samples = 100

    for epoch in range(num_epochs):
        for i, (color_images, b_and_w_images) in enumerate(train_loader):
            minibatch = color_images.size(0)

            # damaged = make_damaged(images)
            # damaged = Variable(damaged)
            color_images = Variable(color_images)
            b_and_w_images = Variable(b_and_w_images)
            labels_1 = Variable(torch.ones(minibatch))
            labels_0 = Variable(torch.zeros(minibatch))

            if use_cuda:
                color_images, labels_0, labels_1 = color_images.cuda(), labels_0.cuda(), labels_1.cuda()#, damaged.cuda()

            # Generator training
            generated_images = generator(b_and_w_images)
            out = discriminator(generated_images)
            loss_img = criterion_MSE(generated_images, color_images)
            loss_1 = criterion_BCE(out, labels_1)
            g_loss = loss_img + loss_1
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

        test_images_bw = Variable(next(iter(test_loader))[0])

        if use_cuda:
            test_noise = test_noise.cuda()

        test_images_colour = generator(test_images_bw)
        test_images_colour =  test_images_colour.view(num_of_samples, 28, 28).data.cpu().numpy()
        filename = "/output/epoch_{}.png" if use_cuda else "samples/epoch_{}.png"
        save_images(test_images_colour, filename=filename.format(epoch + 1), width=10)


if __name__ == "__main__":
    train_GAN(True)