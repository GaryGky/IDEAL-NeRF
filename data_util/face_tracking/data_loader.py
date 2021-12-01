import os
import pickle

import numpy as np
import torch


def load_dir(path, start, end, cuda_num=0):
    lmss = []
    imgs_paths = []
    for i in range(start, end):
        if os.path.isfile(os.path.join(path, str(i) + '.lms')):
            lms = np.loadtxt(os.path.join(
                path, str(i) + '.lms'), dtype=np.float32)
            lmss.append(lms)
            imgs_paths.append(os.path.join(path, str(i) + '.jpg'))
        if i % 100 == 0:
            print(f"face_tracker: load data: {i}.jpg")
    lmss = np.stack(lmss)

    # np.save(f'dataset/{id}/lmss.npy', lmss)
    # f = open(f'dataset/{id}/imgs_paths.obj', 'wb')
    # pickle.dump(imgs_paths, f)

    return torch.from_numpy(lmss).cuda(cuda_num), imgs_paths


if __name__ == '__main__':
    pass
