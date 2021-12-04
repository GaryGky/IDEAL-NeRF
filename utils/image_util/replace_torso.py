import imageio
import matplotlib.pyplot as plt

pred = imageio.imread('output/paper_model/may_torso_bg_head/May_ud_ch_ExpPose_0.jpg')
parsed = imageio.imread('dataset/May/parsing/0.png')
ori_imgs = imageio.imread('dataset/May/bc.jpg')



pred[275:450, 0:450] = ori_imgs[275:450, 0:450]

# plt.imshow(ori_imgs[head_part])
plt.imshow(pred)
plt.show()
