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
        self.deconv_3 = nn.ConvTranspose2d(32, 3, 4, stride=2)

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


def train_GAN(use_cuda=False, numb_style_images=100):
    path = "/data/" if use_cuda else "/home/dobosevych/Documents/Cats/"
    train_loader = load_data(path, upper_bound=21000)
    test_loader = load_data(path, lower_bound=21000, upper_bound=22000)

    lr = 0.0002
    betas = (0.5, 0.999)
    discriminator = Discriminator()
    generator = Generator()
    vgg = Vgg16(requires_grad=False)
    # style_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Lambda(lambda x: x.mul(255))
    # ])
    # style = Image.open(styleimage)
    # style = style_transform(style)
    # # style = style.repeat(args.batch_size, 1, 1, 1)
    #
    # if use_cuda:
    #     vgg.cuda()
    #     style = style.cuda()
    #
    # style_v = Variable(style)
    # style_v = F.batch_norm(style_v)
    # features_style = vgg(style_v)
    #
    #
    # gram_style = [gram_matrix(y) for y in features_style]

    styles = []
    i = 0
    for (color_img, bw_img) in train_loader:
        if i > numb_style_images:
            break
        if use_cuda:
            vgg.cuda()
            color_img = color_img.cuda()

        color_img_v = Variable(color_img)
        color_img_v = F.batch_norm(color_img_v)
        features_style = vgg(color_img_v)


        gram_style = [gram_matrix(y) for y in features_style]
        styles.append(gram_style)


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
                color_images, b_and_w_images, labels_0, labels_1 = color_images.cuda(), b_and_w_images.cuda(), labels_0.cuda(), labels_1.cuda()#, damaged.cuda()

            # Generator training
            generated_images = generator(b_and_w_images)
            out = discriminator(generated_images)


            styleloss = 0
            # for ft_y, gm_s in zip(gen_img_features, gram_style):
            #     gm_y = gram_matrix(gen_img_features)
            #     styleloss += criterion_MSE(gm_y, gm_s[:n_batch, :, :])  #perceptial loss


            for style_img in styles:
                styleloss += style_loss(style_img, generated_images, vgg, minibatch)



            loss_img = criterion_MSE(generated_images, color_images)
            loss_1 = criterion_BCE(out, labels_1)
            # g_loss = 100 * loss_img + loss_1
            g_loss = 100 * styleloss + loss_1
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

        test_images_color, test_images_bw = next(iter(test_loader))
        test_images_bw = Variable(test_images_bw)

        if use_cuda:
            test_images_bw = test_images_bw.cuda()

        test_images_colored = generator(test_images_bw)
        test_images_colored = test_images_colored.view(num_of_samples, 3, 128, 128).data.cpu().numpy()
        filename_colored = "/output/epoch_{}/colored/sample" if use_cuda else "samples/epoch_{}/colored/sample"
        filename_bw = "/output/epoch_{}/black_and_white/sample" if use_cuda else "samples/epoch_{}/black_and_white/sample"
        filename_color = "/output/epoch_{}/incolor/sample" if use_cuda else "samples/epoch_{}/incolor/sample"

        save_images(test_images_colored, filename=filename_colored.format(epoch + 1), width=10, size=(3, 128, 128))
        save_images(test_images_bw, filename=filename_bw.format(epoch + 1), width=10, size=(3, 128, 128))
        save_images(test_images_color, filename=filename_color.format(epoch + 1), width=10, size=(3, 128, 128))


if __name__ == "__main__":
    train_GAN(True)
