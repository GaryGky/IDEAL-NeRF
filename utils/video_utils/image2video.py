import glob

import cv2
import imageio
import numpy as np
import matplotlib.pyplot as plt

dir = 'output/paper_model/Ours/processed_/V_May_obama_N_ExpPose0_aligned'
img = imageio.imread(f'{dir}/frame_det_00_000001.bmp')
print(img.shape)
#
vid_out = cv2.VideoWriter(f'{dir}/0_aligned.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 25, (img.shape[0], img.shape[1]))

images = glob.glob(f'{dir}/*.bmp')
images.sort()

for img in images:
    img_save = imageio.imread(img)
    vid_out.write(cv2.cvtColor(img_save,cv2.COLOR_BGR2RGB))
vid_out.release()

exit(0)

result = []
for i in range(len(images)):
    result.append(imageio.imread(images[i])[0:340,80:420])
    if len(result) > 2:
        break
    i += 90

result = np.concatenate(result, axis=1)
# imageio.imwrite('./may_driven.jpg', result)
plt.axis('off')
plt.imshow(result)
plt.show()
