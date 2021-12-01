import cv2
import einops
import imageio
import torch

from torchvision import models
from collections import namedtuple


class VGG16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],  std=[0.229, 0.224, 0.225])
        # X=normalize(X)
        X = 0.5 * (X + 1.0)  # map to [0,1]

        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


class VGGLOSS(torch.nn.Module):
    def __init__(self):
        super(VGGLOSS, self).__init__()
        self.model = VGG16()
        self.criterionL2 = torch.nn.MSELoss(reduction='mean')

    def forward(self, fake, target):
        # print(self.model)

        vgg_fake = self.model(fake)
        vgg_target = self.model(target)

        content_loss = self.criterionL2(vgg_target.relu4_3, vgg_fake.relu4_3) + \
                       self.criterionL2(vgg_target.relu3_3, vgg_fake.relu3_3) + \
                       self.criterionL2(vgg_target.relu2_2, vgg_fake.relu2_2) + \
                       self.criterionL2(vgg_target.relu1_2, vgg_fake.relu1_2)

        extra = {
            'fake_feature_map': vgg_fake,
            'target_feature_map': vgg_target
        }
        return content_loss, extra


if __name__ == '__main__':
    vgg_loss = VGGLOSS()
    gt_path = 'output/test_input/10_gt.jpg'
    rgb_path = 'output/test_input/10_rgb.jpg'

    gt_img = imageio.imread(gt_path) / 255.0
    rgb_img = imageio.imread(rgb_path) / 255.0

    gt_img = einops.repeat(torch.tensor(gt_img, requires_grad=True, dtype=torch.float32), 'h w c ->b c h w', b=1)
    rgb_img = einops.repeat(torch.tensor(rgb_img, requires_grad=True, dtype=torch.float32), 'h w c ->b c h w', b=1)

    # print(gt_img.shape)

    loss, extra = vgg_loss(rgb_img, gt_img)

    print(f'vgg_loss:{loss.detach().item()}')

    target_feature_map = extra['fake_feature_map'].relu1_2
    fake_feature_map = extra['fake_feature_map'].relu1_2

    target_feature_map_flatten = torch.squeeze(target_feature_map).detach().cpu().numpy()
    fake_feature_map_flatten = torch.squeeze(fake_feature_map).detach().cpu().numpy()

    print(fake_feature_map_flatten.shape)

    target_feature_maps = []
    fake_feature_maps = []
    import matplotlib.pyplot as plt

    plt.figure(figsize=(32, 16))
    for i in range(target_feature_map_flatten.shape[0]):
        # target_feature_maps.append(target_feature_map_flatten[i])
        fake_feature_maps.append(fake_feature_map_flatten[i])
        plt.subplot(32, 16, i + 1)
        plt.imshow(fake_feature_map_flatten[i])
        plt.axis('off')

    plt.show()
