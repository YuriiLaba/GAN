#COMMON GAN USED FOR IMAGE GENERATION
import torch
import torchvision
import torch.nn as nn
import torchvision.datasets as dsets
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F

from manipulation import save_images

# from manipulation import save_images

# Settings
# torch.set_num_threads(4)

from dataloader import *

DIM = 32


def load_data():
    batch_size = 100

    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    dataset = ImageDataset("/data/Cats_color_32", "/data/Cats_B&W_32", transform=transform)
    # dataset = ImageDataset("/home/yuriy/Cats/Cats_32/Cats_color_32", "/home/yuriy/Cats/Cats_32/Cats_B&W_32", transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


class Discriminator(nn.Module):
    def __init__(self, dim):
        assert dim // 2
        super().__init__()
        # self.fc_1 = nn.Linear(16384, 32768)
        #
        # self.fc_2_bn = nn.BatchNorm2d(0.3),
        # self.fc_3 = nn.Linear(32768, 16384),
        #
        # self.fc_4_bn = nn.BatchNorm2d(0.3),
        # self.fc_5 = nn.Linear(16384, 8192),
        #
        # self.fc_6_bn = nn.BatchNorm2d(0.3),
        # self.fc_7 = nn.Linear(8192, 1),
        self.dim = dim
        self.fc_1 = nn.Linear(dim ** 2, dim ** 2 * 2)
        self.fc_2 = nn.Linear(dim ** 2 * 2, dim ** 2)
        self.fc_2_bn = nn.BatchNorm1d(dim ** 2)

        self.fc_3 = nn.Linear(dim ** 2, dim ** 2 // 2)
        self.fc_3_bn = nn.BatchNorm1d(dim ** 2 // 2)

        self.fc_4 = nn.Linear(dim ** 2 // 2, 1)

    def forward(self, x):
        # print(x.size())
        x = F.leaky_relu(self.fc_1(x), 0.2)
        x = F.leaky_relu(self.fc_2_bn(self.fc_2(x)), 0.2)
        x = F.leaky_relu(self.fc_3_bn(self.fc_3(x)), 0.2)
        out = F.sigmoid(self.fc_4(x))
        out = out.view(out.size(0), -1)
        return out


class Generator(nn.Module):
    def __init__(self, dim):
        assert dim // 2
        super().__init__()
        self.dim = dim
        # self.fc_1 = nn.Linear(1024, 2048),
        # self.fc_2 = nn.Linear(2048, 4096),
        # self.fc_3 = nn.Linear(4096, 8192),
        # self.fc_4 = nn.Linear(8192, 16384)
        self.fc_1 = nn.Linear(100, dim ** 2 // 8)
        self.fc_2 = nn.Linear(dim ** 2 // 8, dim ** 2 // 4)
        self.fc_3 = nn.Linear(dim ** 2 // 4, dim ** 2 // 2)
        self.fc_4 = nn.Linear(dim ** 2 // 2, dim ** 2)

    def forward(self, x):
        x = F.leaky_relu(self.fc_1(x), 0.2)
        x = F.leaky_relu(self.fc_2(x), 0.2)
        x = F.leaky_relu(self.fc_3(x), 0.2)
        out = F.tanh(self.fc_4(x))
        return out


def train_GAN(use_cuda=False):
    # train_dataset = dsets.MNIST(root='./data/', train=True, download=True, transform=transform)
    train_loader = load_data()

    lr = 0.0002
    betas = (0.5, 0.999)
    discriminator = Discriminator(DIM)
    generator = Generator(DIM)

    if use_cuda:
        discriminator = discriminator.cuda()
        generator = generator.cuda()

    # print(len(list(discriminator.parameters())))
    # print(list(generator.parameters()))
    criterion = nn.BCELoss()
    d_optimizer = torch.optim.Adam(list(discriminator.parameters()), lr=lr, betas=betas)
    g_optimizer = torch.optim.Adam(list(generator.parameters()), lr=lr, betas=betas)




    num_epochs = 200
    num_of_samples = 64

    # Training
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



            # Testing our result on each epoch
            test_noise = Variable(torch.randn(num_of_samples, 100))

            if use_cuda:
                test_noise = Variable(torch.randn(num_of_samples, 100)).cuda()

            test_images = generator(test_noise)
            test_images = test_images.view(num_of_samples, DIM, DIM).data.cpu().numpy()
            save_images(test_images, filename="/output/epoch_{}.png".format(epoch + 1))


if __name__ == "__main__":
    train_GAN(False)
