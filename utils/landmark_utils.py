import face_alignment
import imageio
import numpy as np
import torch
import cv2
import os

id = 'Chinese'
ori_imgs_dir = os.path.join('dataset', id, 'ori_imgs')


def get_lms_features(lms):
    feature_lms = []
    for i in range(0, len(lms)):
        base = lms[i]
        for j in range(i + 1, len(lms)):
            comp = lms[j]
            dis = torch.sqrt(torch.sum((base - comp) ** 2))
            feature_lms.append(dis)

    feature_lms = torch.as_tensor(feature_lms)
    feature_lms = feature_lms / max(feature_lms)
    return torch.Tensor(feature_lms)


def get_lms_features_np(lms):
    feature_lms = []
    for i in range(0, len(lms)):
        base = lms[i]
        for j in range(i + 1, len(lms)):
            comp = lms[j]
            dis = np.sqrt(np.sum((base - comp) ** 2))
            feature_lms.append(dis)

    feature_lms = feature_lms / np.max(feature_lms)
    return feature_lms


def get_relative_distance():
    for i in range(10000):
        lms_path = os.path.join(ori_imgs_dir, f'{i}.lms')
        if os.path.isfile(lms_path):
            lms = np.loadtxt(lms_path)
            lms_feature = get_lms_features_np(lms)
            np.savetxt(lms_path[:-3] + 'lf', lms_feature)
            if i % 5 == 0:
                print(f'save: {i} landmarks successful')


def face_location():
    import face_recognition
    for i in range(160, 1000):
        img_path = f'dataset/Noah/ori_imgs/{i}.jpg'

        img = face_recognition.load_image_file(img_path)

        preds = face_recognition.face_locations(img)
        print(preds)

        if len(preds) == 0:
            print(f'image:{i} can not detect face!!')
            continue

        x1, y1, x2, y2 = int(preds[0][0]), int(preds[0][1]), int(preds[0][2]), int(preds[0][3])

        cv2.imwrite(img_path, img[x1 - 132:x2 + 132, y2 - 132:y1 + 132])

        if i % 10 == 0:
            print(f'save {i} img done')


def draw_landmark():
    import matplotlib.pyplot as plt

    contour = np.arange(0, 17)
    left_eyebrow, right_eyebrow = np.arange(17, 22), np.arange(22, 27)
    nose, mouth = np.arange(27, 36), np.arange(48, 68)
    left_eyes, right_eyes = np.arange(36, 42), np.arange(42, 48)

    landmark_path = 'dataset/Noah/ori_imgs/0.lms'
    landmark = np.loadtxt(landmark_path)

    landmark_idx = [contour, left_eyes, right_eyes, left_eyebrow, right_eyebrow, nose, mouth]

    ax = plt.gca()
    for idx in landmark_idx:
        ax.plot(landmark[idx, 0], 450 - landmark[idx, 1], color='y', linewidth=3, alpha=.6)
        ax.axis('off')
    plt.show()

    img = imageio.imread('dataset/Noah/ori_imgs/0.jpg')
    for pts in landmark:
        print(pts)
        cv2.circle(img, (int(pts[0]), int(pts[1])), radius=3, color=(0, 0, 255), thickness=-1)
        cv2.imshow('test', img)
        cv2.waitKey(10000)


def detect_lands(ori_imgs_dir):
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cuda')
    start = 0

    for image_path in os.listdir(ori_imgs_dir):
        if image_path.endswith('.jpg'):
            input = imageio.imread(os.path.join(ori_imgs_dir, image_path))[:, :, :3]
            preds = fa.get_landmarks(input)
            if start % 100 == 0:
                print(f'finish: {start} landmark detect!')
            if len(preds) > 0:
                lands = preds[0].reshape(-1, 2)[:, :2]
                # lf = get_lms_features_np(preds[0])
                np.savetxt(os.path.join(ori_imgs_dir, image_path[:-3] + 'lms'), lands, '%f')
                # np.savetxt(os.path.join(ori_imgs_dir, image_path[:-3] + 'lf'), lf, '%f')
        start += 1


if __name__ == '__main__':
    import configargparse

    parser = configargparse.ArgumentParser()
    parser.add_argument('--dir', type=str, help='ori_imgs path')

    args = parser.parse_args()
    print(f'ori_imgs path: {args.dir}')
    detect_lands(ori_imgs_dir=args.dir)
