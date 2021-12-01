import einops
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F


class VGGFace(nn.Module):
    def __init__(self):
        super(VGGFace, self).__init__()
        # in_ch, out_ch, kernel_size
        self.conv1_1 = nn.Conv2d(3, 64, 3)
        self.conv1_2 = nn.Conv2d(64, 64, 3)
        self.conv2_1 = nn.Conv2d(64, 128, 3)
        self.conv2_2 = nn.Conv2d(128, 128, 3)
        self.conv3_1 = nn.Conv2d(128, 256, 3)
        self.conv3_2 = nn.Conv2d(256, 256, 3)
        self.conv3_3 = nn.Conv2d(256, 256, 3)
        self.conv4_1 = nn.Conv2d(256, 512, 3)
        self.conv4_2 = nn.Conv2d(512, 512, 3)
        self.conv4_3 = nn.Conv2d(512, 512, 3)
        self.conv5_1 = nn.Conv2d(512, 512, 3)

    def forward(self, x):
        conv1_1_pad = F.pad(x, (1, 1, 1, 1))
        conv1_1 = self.conv1_1(conv1_1_pad)
        relu1_1 = F.relu(conv1_1)

        conv1_2_pad = F.pad(relu1_1, (1, 1, 1, 1))
        conv1_2 = self.conv1_2(conv1_2_pad)
        relu1_2 = F.relu(conv1_2)

        pool1_pad = F.pad(relu1_2, (0, 1, 0, 1), value=float('-inf'))
        pool1 = F.max_pool2d(pool1_pad, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)

        conv2_1_pad = F.pad(pool1, (1, 1, 1, 1))
        conv2_1 = self.conv2_1(conv2_1_pad)
        relu2_1 = F.relu(conv2_1)

        conv2_2_pad = F.pad(relu2_1, (1, 1, 1, 1))
        conv2_2 = self.conv2_2(conv2_2_pad)
        relu2_2 = F.relu(conv2_2)

        pool2_pad = F.pad(relu2_2, (0, 1, 0, 1), value=float('-inf'))
        pool2 = F.max_pool2d(pool2_pad, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)

        conv3_1_pad = F.pad(pool2, (1, 1, 1, 1))
        conv3_1 = self.conv3_1(conv3_1_pad)
        relu3_1 = F.relu(conv3_1)

        conv3_2_pad = F.pad(relu3_1, (1, 1, 1, 1))
        conv3_2 = self.conv3_2(conv3_2_pad)
        relu3_2 = F.relu(conv3_2)

        conv3_3_pad = F.pad(relu3_2, (1, 1, 1, 1))
        conv3_3 = self.conv3_3(conv3_3_pad)
        relu3_3 = F.relu(conv3_3)

        pool3_pad = F.pad(relu3_3, (0, 1, 0, 1), value=float('-inf'))
        pool3 = F.max_pool2d(pool3_pad, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)

        conv4_1_pad = F.pad(pool3, (1, 1, 1, 1))
        conv4_1 = self.conv4_1(conv4_1_pad)
        relu4_1 = F.relu(conv4_1)

        conv4_2_pad = F.pad(relu4_1, (1, 1, 1, 1))
        conv4_2 = self.conv4_2(conv4_2_pad)
        relu4_2 = F.relu(conv4_2)

        conv4_3_pad = F.pad(relu4_2, (1, 1, 1, 1))
        conv4_3 = self.conv4_3(conv4_3_pad)
        relu4_3 = F.relu(conv4_3)

        pool4_pad = F.pad(relu4_3, (0, 1, 0, 1), value=float('-inf'))
        pool4 = F.max_pool2d(pool4_pad, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)

        conv5_1_pad = F.pad(pool4, (1, 1, 1, 1))
        conv5_1 = self.conv5_1(conv5_1_pad)
        relu5_1 = F.relu(conv5_1)

        return [relu1_1, relu2_1, relu3_1, relu4_1, relu5_1]


class VGGFaceLoss(nn.Module):
    def __init__(self, vgg_face_weight='loss/vgg_face.pth'):
        super(VGGFaceLoss, self).__init__()
        self.model = VGGFace()
        weight = torch.load(vgg_face_weight)
        self.model.load_state_dict(weight, strict=False)
        self.model.eval()

    def forward(self, fake, target):
        fake_features = self.model(fake)
        target_features = self.model(target)

        content_loss = torch.tensor(0.)
        for fake_feat, target_feat in zip(fake_features, target_features):
            content_loss += F.l1_loss(fake_feat, target_feat)

        return content_loss, {'fake': fake_features, 'target': target_features}


if __name__ == '__main__':
    target_path, fake_path = 'output/test_input/0_gt.jpg', 'output/test_input/0_rgb.jpg'
    vgg_face = VGGFace()

    vggFaceLoss = VGGFaceLoss()
    fake, target = imageio.imread(fake_path) / 255.0, imageio.imread(target_path) / 255.0

    fake, target = torch.tensor(fake, dtype=torch.float32, requires_grad=True), torch.tensor(target,
                                                                                             dtype=torch.float32,
                                                                                             requires_grad=True)

    fake, target = einops.repeat(fake, 'h w c -> b c h w', b=1), einops.repeat(target, 'h w c -> b c w h', b=1)

    vgg_face_loss, type2featureMap = vggFaceLoss(fake, target)
    print(vgg_face_loss)

    # import matplotlib.pyplot as plt
    #
    # plt.figure(figsize=(8, 8))
    # fake_features = type2featureMap['fake'][4]
    #
    # fake_features_squeeze = torch.squeeze(fake_features).detach().cpu().numpy()
    # for i in range(fake_features_squeeze.shape[0]):
    #     plt.subplot(16, 32, i + 1)
    #     plt.imshow(fake_features_squeeze[i])
    #     plt.axis('off')
    #
    # plt.show()
