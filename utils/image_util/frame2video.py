import os

import cv2
import imageio

if __name__ == '__main__':
    frame_path = 'dataset/Obama/ori_imgs'
    save_path = 'output/cross_subject/May-Obama-Expr'
    vid_out = cv2.VideoWriter(os.path.join(save_path, 'Obama.avi'),
                              cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 25, (450, 450))
    for i in range(5521, 5521 + 553):
        img = imageio.imread(f'{frame_path}/{i}.jpg')
        vid_out.write(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    vid_out.release()
