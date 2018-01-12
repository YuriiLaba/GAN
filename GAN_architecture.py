import torch
import torchvision
import torch.nn as nn
import torchvision.datasets as dsets
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F

# from manipulation import save_images

# Settings
# torch.set_num_threads(4)

from dataloader import *



def load_data():
    batch_size = 100

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    dataset = ImageDataset("/home/yuriy/Cats/Cats_color", "/home/yuriy/Cats/Cats_B&W")
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(len(dataset.images_color_lst))
    print(len(dataset.images_b_and_w_lst))

    return loader



class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_1 = nn.Linear(16384, 32768)

        self.fc_2_bn = nn.BatchNorm2d(0.3),
        self.fc_3 = nn.Linear(32768, 16384),

        self.fc_4_bn = nn.BatchNorm2d(0.3),
        self.fc_5 = nn.Linear(16384, 8192),

        self.fc_6_bn = nn.BatchNorm2d(0.3),
        self.fc_7 = nn.Linear(8192, 1),



    def forward(self, x):
        x = F.leaky_relu(self.fc_1(x), 0.2)
        x = F.leaky_relu(self.fc_3(self.fc_2_bn(x)), 0.2)
        x = F.leaky_relu(self.fc_5(self.fc_4_bn(x)), 0.2)
        out = F.sigmoid(self.fc_7(self.fc_6_bn(x)))

        out = out.view(out.size(0), -1)
        return out


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_1 = nn.Linear(1024, 2048),
        self.fc_2 = nn.Linear(2048, 4096),
        self.fc_3 = nn.Linear(4096, 8192),
        self.fc_4 = nn.Linear(8192, 16384),


    def forward(self, x):
        x = x.view(x.size(0), 100)
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
    discriminator = Discriminator()
    generator = Generator()

    criterion = nn.BCELoss()
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=betas)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=betas)


    if use_cuda:
        discriminator = discriminator.cuda()
        generator = generator.cuda()



    num_epochs = 200
    num_of_samples = 64

    # Training
    for epoch in range(num_epochs):
        for i, (color_images, b_and_w_images) in enumerate(train_loader):
            true_images = Variable(b_and_w_images.view(-1, 128 * 128)) # discriminator input is 28 * 28
            true_labels = Variable(torch.ones(true_images.size(0)))


            # Sample data from generator
            noise = Variable(torch.randn(true_images.size(0), 1024)) # generator input is 100
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
            noise = Variable(torch.randn(b_and_w_images.size(0), 1024))
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
    # test_noise = Variable(torch.randn(num_of_samples, 100)).cuda()
    # test_images = generator(test_noise)
    # test_images = test_images.view(num_of_samples, 128, 128).data.cpu().numpy()
    # save_images(test_images, filename="epoch_{}.png".format(epoch + 1))


if __name__ == "__main__":
    train_GAN(False)