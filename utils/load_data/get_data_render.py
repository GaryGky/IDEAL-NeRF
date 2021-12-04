import json
import os

import cv2
import einops
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
    def __init__(self, data_dir, aud_file, mode, args, skip=1):
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
        self.aud_features = np.load(os.path.join(self.data_dir, aud_file))
        self.background_img = torch.tensor(imageio.imread(os.path.join('dataset/Chinese', 'bc_450.jpg')) / 255.0)
        self.all_landmarks = []
        self.landmark_features = []
        self.lms_shape = 68

        self.skip = 1 if self.mode == "train" else args.testskip

        for frame in self.meta['frames'][::skip]:
            fname = os.path.join(self.data_dir, 'head_imgs', str(frame['img_id']) + '.jpg')
            landmark = os.path.join(self.data_dir, 'ori_imgs', str(frame['img_id']) + '.lms')
            landmark_feature = os.path.join(self.data_dir, 'ori_imgs', str(frame['img_id']) + '.lf')

            self.all_landmarks.append(landmark)
            self.landmark_features.append(landmark_feature)
            self.all_imgs.append(fname)
            self.all_poses.append(np.array(frame['transform_matrix']))
            self.auds.append(self.aud_features[min(frame['aud_id'], self.aud_features.shape[0] - 1)])
            self.all_face_rects.append(np.array(frame['face_rect'], dtype=np.int32))  # 人脸的Bbox

        self.data_size = min(len(self.aud_features), len(self.all_poses))
        self.focal, self.cx, self.cy = float(self.meta['focal_len']), float(self.meta['cx']), float(self.meta['cy'])
        self.H, self.W = self.cy * 2, self.cx * 2
        self.args = args

        # 加载driven landmark
        self.target_landmarks = []
        for i in range(50):
            lms_path = os.path.join('dataset/trump/ori_imgs', f'{i}.lf')
            if os.path.isfile(lms_path):
                lms = np.loadtxt(lms_path)
                self.target_landmarks.append(lms)

    def __getitem__(self, index):
        """ 支持下标索引，通过index把dataset中的数据拿出来"""
        if index is None:
            """ 未传下标时使用随机的方法采样 """
            index = np.random.choice(self.data_size)

        # 目标图像 + 人脸检测
        image = face_recognition.load_image_file(self.all_imgs[index])
        face_rects = face_recognition.face_locations(image)
        rect = face_rects[0]
        self.H, self.W = image.shape[0], image.shape[1]

        target = image[rect[0] - 50:rect[2] + 50, rect[3] - 50:rect[1] + 50]
        pose = self.all_poses[index][:3, :4]

        # 裁剪出人脸的部分
        bc_rgb = self.background_img[rect[0] - 50:rect[2] + 50, rect[3] - 50:rect[1] + 50].cuda()
        target = torch.tensor(target).cuda().float() / 255.0

        auds = torch.tensor(self.auds, dtype=torch.float).cuda()
        pose = torch.tensor(pose).cuda()
        index = torch.tensor(index).cuda()

        return torch.tensor(0), torch.tensor(0), bc_rgb, auds, target, pose, torch.tensor(0), index

    def __len__(self):
        """返回数据集条数"""
        return self.data_size


if __name__ == '__main__':
    def to8b(x): return (255 * np.clip(x, 0, 1)).astype(np.uint8)


    parser = config_parser()
    args = parser.parse_args()

    dataset_train = GetData('dataset/Obama', 'aud.npy', mode="train", args=args)
    train_data = dataset_train.__getitem__(0)
    targets = train_data[1]
    targets = einops.rearrange(targets, '(h w) c -> h w c', h=64)
    targets_cpu = targets.cpu().numpy()
    cv2.cvtColor(targets_cpu, cv2.COLOR_BGR2RGB)

    cv2.imwrite('test_target.jpg', to8b(targets_cpu))

    # print(np.sum(train_data[-2] - train_data[-3]))
