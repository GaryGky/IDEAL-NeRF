import cv2
import numpy as np
import matplotlib.pyplot as plt


def to8b(x): return (255 * np.clip(x, 0, 1)).astype(np.uint8)


noise = np.random.randint(0, 100, (8, 29, 1))
print(noise.shape)

# aud_npy = np.load('dataset/Chinese/aud_ch.npy')
# aud_vis = aud_npy[0][8:] + noise
#
# aud_vis = aud_vis[..., np.newaxis]
# print(aud_vis.shape)
#
# aud_ch3 = np.concatenate([aud_vis, aud_vis, aud_vis], axis=2)
# aud_ch3 = to8b(aud_ch3 - np.min(aud_ch3))
# aud_ch3[..., 0] = aud_ch3[..., 0] - 30
# aud_ch3[..., 1] = aud_ch3[..., 1] - 10
# aud_ch3[..., 2] = aud_ch3[..., 2] - 20

plt.axis('off')
plt.imshow(noise)
plt.show()
