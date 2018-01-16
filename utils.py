import torch.nn.functional as F
from dataloader import *
import torch.nn as nn
from torch.autograd import Variable
# from torch.optim import Adam
from torchvision import transforms
import torchvision.datasets as dsets


from vgg import Vgg16
from manipulation import save_images



def style_loss(style_img_gram_matrices, gen_img, vgg, minibatch):
    styleloss = 0
    gen_img_n = F.batch_norm(gen_img)
    gen_img_features = vgg(gen_img_n)
    for ft_y, gm_s in zip(gen_img_features, style_img_gram_matrices):
        gm_y = gram_matrix(gen_img_features)
        styleloss += nn.MSELoss(gm_y, gm_s[:minibatch, :, :])  # perceptial loss
    return styleloss



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


def get_gram_matrices(images):


    # for (color_img, bw_img) in imgs:
    #     if use_cuda:
    #         color_img = color_img.cuda()

        # color_img_v = Variable(color_img)
    images = F.batch_norm(images)
    features_style = vgg(images)
    gram_style = [gram_matrix(y) for y in features_style]
    styles.append(gram_style)
    return styles