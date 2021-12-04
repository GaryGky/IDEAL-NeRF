import json
import os

import cv2
import face_recognition
import imageio
import numpy as np
import torch
from torch.multiprocessing import Pool, Process, set_start_method
from torch.utils.data import Dataset  # Dataset是个抽象类，只能用于继承

from NeRFs.HeadNeRF.helper import config_parser

try:
    set_start_method('spawn')
except RuntimeError:
    pass


class GetData(Dataset):
    # mode is [train, val, test]
    # 从Noah中需要获取：Landmark Audio Pose
    # 从Obama中需要获取：Background HeadImage
    def __init__(self, source_dir, data_dir, aud_file, mode, args, skip=1):
        self.data_dir = data_dir
        self.aud_file = aud_file
        self.mode = mode
        self.meta = None
        with open(os.path.join(data_dir, 'transforms_{}.json'.format(mode)), 'r') as fp:
            self.meta = json.load(fp)
        self.all_imgs = []
        self.all_poses = []
        self.auds = []
        self.all_face_rects = []
        # 使用Noah的音频
        self.aud_features = np.load(os.path.join(source_dir, aud_file))
        self.background_img = imageio.imread(os.path.join(self.data_dir, 'bc.jpg')) / 255
        self.all_landmarks = []
        self.landmark_features = []
        self.lms_shape = 68

        self.skip = 1 if self.mode == "train" else args.testskip

        for frame in self.meta['frames'][::skip]:
            if frame['img_id'] > len(self.aud_features):
                break

            # 获取Obama头部的图像和pose
            fname = os.path.join(self.data_dir, 'head_imgs', str(frame['img_id']) + '.jpg')
            pose = np.array(frame['transform_matrix'])
            self.all_imgs.append(fname)
            self.all_poses.append(pose)

            # 获取Noah的人脸关键点和audio
            landmark = os.path.join(source_dir, 'ori_imgs', str(frame['img_id']) + '.lms')
            landmark_feature = os.path.join(source_dir, 'ori_imgs', str(frame['img_id']) + '.lf')
            aud = self.aud_features[min(frame['aud_id'], self.aud_features.shape[0] - 1)]
            self.auds.append(aud)
            self.all_landmarks.append(landmark)
            self.landmark_features.append(landmark_feature)

        self.data_size = min(len(self.all_imgs), len(self.auds))
        self.focal, self.cx, self.cy = float(self.meta['focal_len']), float(self.meta['cx']), float(self.meta['cy'])
        self.args = args

    def __getitem__(self, index):
        """ 支持下标索引，通过index把dataset中的数据拿出来"""
        if index is None:
            """ 未传下标时使用随机的方法采样 """
            index = np.random.choice(self.data_size)

        # 目标图像
        raw_img = imageio.imread(self.all_imgs[index]) / 255
        self.H, self.W = raw_img.shape[0], raw_img.shape[1]
        # 这里target只是用来比较
        pose = self.all_poses[index][:3, :4]

        # 第二阶段：直接采样整张图片会导致OOM，所以尝试采样Obama人脸部分
        image = face_recognition.load_image_file(self.all_imgs[index])
        face_rects = face_recognition.face_locations(image)
        face_rect = face_rects[0]

        auds = torch.tensor(self.auds, dtype=torch.float)  # audio 特征

        target_lf = np.loadtxt(self.landmark_features[index])  # Noah的人脸关键点特征
        target_lf = torch.tensor(target_lf)

        # 裁剪出人脸的部分
        bc_rgb = self.background_img[face_rect[0]:face_rect[2], face_rect[3]:face_rect[1]]
        gt_image = raw_img[face_rect[0]:face_rect[2], face_rect[3]:face_rect[1]]

        bc_rgb = cv2.resize(bc_rgb, (225, 225), interpolation=cv2.INTER_NEAREST)
        gt_image = cv2.resize(gt_image, (225, 225), interpolation=cv2.INTER_NEAREST)

        return torch.tensor(0), torch.tensor(0), bc_rgb, auds, gt_image, pose, target_lf, index

    def __len__(self):
        """返回数据集条数"""
        return self.data_size

    def get_rays(self, H, W, focal, c2w, cx=None, cy=None):
        i, j = torch.meshgrid(torch.linspace(0, W - 1, W, requires_grad=False),
                              torch.linspace(0, H - 1, H, requires_grad=False))
        i = i.t()
        j = j.t()
        if cx is None:
            cx = W * .5
        if cy is None:
            cy = H * .5
        dirs = torch.stack(
            [(i - cx) / focal, -(j - cy) / focal, -torch.ones_like(i)], -1).to(c2w.device)
        # Rotate ray directions from camera frame to the world frame
        # dot product, equals to: [c2w.dot(dir) for dir in dirs]
        rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
        # Translate camera frame's origin to the world frame. It is the origin of all rays.
        rays_o = c2w[:3, -1].expand(rays_d.shape)
        return rays_o, rays_d

    def sample_rays(self, pose, face_rect, target, bc_img, landmark):
        rays_o, rays_d = self.get_rays(self.H, self.W, self.focal, torch.Tensor(pose).cuda(), self.cx,
                                       self.cy)  # (H, W, 3), (H, W, 3)

        coords = torch.stack(
            torch.meshgrid(torch.linspace(0, self.H - 1, self.H),
                           torch.linspace(0, self.W - 1, self.W)), -1)  # (H, W, 2)

        coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)

        # 这里是计算人脸的位置
        rect_inds = (coords[:, 0] >= face_rect[0]) & (coords[:, 0] <= face_rect[0] + face_rect[2]) \
                    & (coords[:, 1] >= face_rect[1]) & (coords[:, 1] <= face_rect[1] + face_rect[3])

        coords_rect = coords[rect_inds]  # 包含人脸的像素
        coords_norect = coords[~rect_inds]  # 不包含人脸的像素

        """ 计算除了关键点之外的采样数量 """
        sample_num = self.args.N_rand - self.lms_shape
        rect_num = int(sample_num * self.args.sample_rate)
        norect_num = sample_num - rect_num

        select_inds_rect = np.random.choice(coords_rect.shape[0], size=[rect_num], replace=False)  # (N_rand,)
        select_coords_rect = coords_rect[select_inds_rect].long()  # (N_rand * sample_rate, 2)

        select_inds_norect = np.random.choice(
            coords_norect.shape[0], size=[norect_num], replace=False)  # (N_rand,)
        select_coords_norect = coords_norect[select_inds_norect].long()  # (N_rand * (1-sample_rate), 2)

        select_coords = torch.cat((torch.tensor(landmark), select_coords_rect, select_coords_norect), dim=0).long()

        rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        batch_rays = torch.stack([rays_o, rays_d], 0)
        target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        bc_rgb = bc_img[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

        return batch_rays, target_s, bc_rgb


if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()

    dataset_train = GetData('dataset/Obama', 'aud.npy', mode="train", args=args)
    train_data = dataset_train.__getitem__(0)

    print(np.sum(train_data[-2] - train_data[-3]))
