import glob
import json
import os

import imageio
import numpy as np


def load_audface_data(datadir, testskip=1, test_file=None, aud_file=None, head_nerf=True,
                      lms_file="/home/pusuan.wk/gky/AD-NeRF/dataset/xidada/ori_imgs"):
    # 测试
    if test_file is not None:
        with open(os.path.join(datadir, test_file)) as fp:
            meta = json.load(fp)
        poses = []
        auds = []
        aud_features = np.load(os.path.join(datadir, aud_file))
        for frame in meta['frames'][::testskip]:
            poses.append(np.array(frame['transform_matrix']))
            auds.append(aud_features[min(frame['aud_id'], len(aud_features)-1)])
        poses = np.array(poses).astype(np.float32)
        auds = np.array(auds).astype(np.float32)
        bc_img = imageio.imread(os.path.join(datadir, 'bc.jpg'))
        H, W = bc_img.shape[0], bc_img.shape[1]
        focal, cx, cy = float(meta['focal_len']), float(
            meta['cx']), float(meta['cy'])
        return poses, auds, bc_img, [H, W, focal, cx, cy]

    splits = ['train', 'val']
    metas = {}
    for s in splits:
        with open(os.path.join(datadir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    all_auds = []
    all_sample_rects = []
    aud_features = np.load(os.path.join(datadir, aud_file))
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        auds = []
        face_rects = []
        mouth_rects = []
        # exps = []
        if s == 'train' or testskip == 0:
            skip = 1
        else:
            skip = testskip

        for frame in meta['frames'][::skip]:
            fname = os.path.join(datadir, 'head_imgs', str(frame['img_id']) + '.jpg')
            imgs.append(fname)
            poses.append(np.array(frame['transform_matrix']))
            auds.append(aud_features[min(frame['aud_id'], aud_features.shape[0] - 1)])
            # 人脸的Bbox
            face_rects.append(np.array(frame['face_rect'], dtype=np.int32))
        imgs = np.array(imgs)
        poses = np.array(poses).astype(np.float32)
        auds = np.array(auds).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
        all_auds.append(auds)
        all_sample_rects.append(face_rects)

    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(len(splits))]
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    auds = np.concatenate(all_auds, 0)
    face_rects = np.concatenate(all_sample_rects, 0)
    bc_img = imageio.imread(os.path.join(datadir, 'bc.jpg'))

    # 如果训练Head，则加载关键点作为监督
    drive_landmarks = []
    if head_nerf:
        all_lms = glob.glob(os.path.join(lms_file, "*.lms"))
        for lms in all_lms:
            drive_landmarks.append(np.loadtxt(lms))

    H, W = bc_img.shape[:2]
    focal, cx, cy = float(meta['focal_len']), float(meta['cx']), float(meta['cy'])
    # 不考虑相机的畸变 可以使用焦距计算内参矩阵
    intrinsic = np.array([[focal, 0., W / 2], [0, focal, H / 2], [0, 0, 1.]])

    return imgs, poses, auds, bc_img, [H, W, focal, cx, cy, intrinsic], \
           face_rects, mouth_rects, i_split, drive_landmarks


if __name__ == '__main__':
    datadir = "/home/pusuan.wk/gky/AD-NeRF/dataset/Obama"
    imgs, poses, auds, bc_img, [H, W, focal, cx, cy], \
    sample_rects, mouth_rects, i_split, drive_landmarks = load_audface_data(datadir, aud_file='xidada.npy',
                                                                            head_nerf=True)
    print(i_split)
