import os
import imageio
import matplotlib.pyplot as plt
import cv2

video_path = 'output/paper_model/Obama3_Obama3_Attention/'
video_name = 'audio.mp4'
type = 'obama'
video_capture = cv2.VideoCapture(os.path.join(video_path, video_name))

os.makedirs(os.path.join(video_path, type), exist_ok=True)

i = 0
# vid_out = cv2.VideoWriter(f'{video_path}/output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 25, (450, 450))
while True:
    ok, frame = video_capture.read()
    # [0:350, 50:400]
    # plt.imshow(frame[0:350, 50:400])
    # plt.show()
    # break
    if not ok:
        break
    if 0 == 0:
        # [0:350, 50:400]
        cv2.imwrite(os.path.join(video_path, f'{type}','{:05d}.jpg'.format(i)), frame)
        # vid_out.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    i += 1
# vid_out.release()
# img = imageio.imread('output/cross_subject_blend/Obama0_Chn_Expr/frame/frame50.jpg')

