import glob

import imageio
import numpy as np
import matplotlib.pyplot as plt

image_path = 'output/paper_model/Ours/frame'
images = glob.glob(f'{image_path}/*.jpg')
images.sort()

print(images)

result = []
for img in images:
    input = imageio.imread(img)[0:400, 30:430]
    result.append(input)
    imageio.imwrite(img, input)
    if len(result) >= 100000:
        break
result = np.concatenate(result, axis=1)

plt.axis('off')
plt.imshow(result)
plt.show()
