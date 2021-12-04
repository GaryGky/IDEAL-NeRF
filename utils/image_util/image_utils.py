import os

import cv2
# import dlib
import face_alignment
# import face_recognition
import imageio
import numpy as np
import torch


def black_out():
    bg_path = 'dataset/Obama/bc.jpg'
    bg_imgs = cv2.imread(bg_path)
    for i in range(5000):
        head_path = f'dataset/Obama/head_imgs/{i}.jpg'
        parse_path = f'dataset/Obama/parsing/{i}.png'

        parse_img = cv2.imread(parse_path)
        head_img = cv2.imread(head_path)

        head_part = head_part = (parse_img[:, :, 0] == 255) & (
                parse_img[:, :, 1] == 0) & (parse_img[:, :, 2] == 0)

        head_img[~head_part] = 0

        cv2.imwrite(head_path, head_img)

        if i % 100 == 0:
            print(f"Save {i} Done!")


def face_location():
    head_path = f'output/test_rgb.jpg'
    head_img = cv2.imread(head_path)
    import face_recognition

    face_rect = face_recognition.face_locations(head_img)[0]
    print(face_rect)

    head_rect = cv2.rectangle(head_img, (face_rect[0], face_rect[1]), (face_rect[2], face_rect[3]), color=(0, 0, 255),
                              thickness=1)

    cv2.imshow('test', head_rect)
    cv2.waitKey(10000)


def shape_to_np(shape, dtype="int"):  # 将包含68个特征的的shape转换为numpy array格式
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def crop_human():
    for i in range(1000, 2000):
        img_path = f'dataset/Chinese/ori_imgs/{i}.jpg'
        try:
            img = face_recognition.load_image_file(img_path)
            if img.shape[0] == 255 and img.shape[1] == 255:
                continue
        except:
            raise None

        img = cv2.resize(img, dsize=None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)

        rect = face_recognition.face_locations(img)[0]
        print(f'{i} image, rect:{rect}')

        tx = (225 - (rect[2] - rect[0])) // 2
        ty = (225 - (rect[1] - rect[3])) // 2

        img_save = img[rect[0] - tx:rect[2] + tx, rect[3] - ty:rect[1] + ty]
        cv2.imwrite(img_path, cv2.cvtColor(img_save, cv2.COLOR_BGR2RGB))


def detect_landmark():
    import face_alignment

    img_path = 'dataset/Obama/ori_imgs/0.jpg'
    face_detector = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cpu')
    pred = face_detector.get_landmarks(imageio.imread(img_path))
    print(pred)


def detect_with_dlib():
    import face_alignment

    img_path = 'output/Obama/noah/Noah_0.jpg'

    img = torch.tensor(face_recognition.load_image_file(img_path), requires_grad=True)
    rects = face_recognition.face_locations(img)
    face_detector = dlib.shape_predictor("utils/shape_predictor_68_face_landmarks.dat")

    rect = list(rects[0])
    landmark = shape_to_np(
        face_detector(img, dlib.rectangle(left=rect[0], top=rect[3], right=rect[2], bottom=rect[1])))

    for pts in landmark:
        cv2.circle(img, (int(pts[0]), int(pts[[1]])), radius=3, color=(255, 0, 0), thickness=-1)

    cv2.imshow('1', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    cv2.waitKey(10000)


def crop_image_with_face():
    head_path = f'output/0.jpg'
    head_img = cv2.imread(head_path)
    import face_recognition

    face_rect = face_recognition.face_locations(head_img)[0]
    print(face_rect)

    len_x, len_y = face_rect[2] - face_rect[0], face_rect[1] - face_rect[3]

    fill_x, fill_y = 224 - len_x, 224 - len_y

    left, top = fill_x // 2, fill_y // 2
    right, bottom = fill_x - left, fill_y - top

    print(left, right, top, bottom)

    head_img = head_img[face_rect[0] - left:face_rect[2] + right, face_rect[3] - bottom:face_rect[1] + top]

    cv2.imshow('test', head_img)
    cv2.waitKey(10000)


def crop_mouth_region():
    img = imageio.imread('dataset/Obama/ori_imgs/0.jpg')
    landmark = np.loadtxt('dataset/Obama/ori_imgs/0.lms')
    landmark_mouth = landmark[48:]

    max_x, min_x = np.max(landmark_mouth[:, 0]) + 20, np.min(landmark_mouth[:, 0]) - 20
    max_y, min_y = np.max(landmark_mouth[:, 1]) + 20, np.min(landmark_mouth[:, 1]) - 20

    print(f'min_x={min_x}, min_y={min_y}, max_x={max_x}, max_y={max_y}')
    for pts in landmark_mouth:
        cv2.circle(img, (int(pts[0]), int(pts[1])), radius=2, color=(100, 150, 30), thickness=-1)
    cv2.rectangle(img, (int(min_x), int(min_y)), (int(max_x), int(max_y)), color=(70, 20, 20), thickness=1)
    cv2.imshow('Obama', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    cv2.waitKey(100000)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    img = imageio.imread('utils/image_util/image/PassportPhoto.jpg')
    img = cv2.resize(img, (400, 514), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite('utils/image_util/image/514_400.jpg', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    print(img.shape)
