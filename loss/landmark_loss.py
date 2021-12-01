import warnings
from distutils.version import LooseVersion
from enum import IntEnum

import imageio
import torch
import torch.nn.functional as F
from face_alignment.utils import *
from torch import nn


class LandmarksType(IntEnum):
    _2D = 1
    _2halfD = 2
    _3D = 3


class NetworkSize(IntEnum):
    LARGE = 4


default_model_urls = {
    '2DFAN-4': 'https://www.adrianbulat.com/downloads/python-fan/2DFAN4-cd938726ad.zip',
    '3DFAN-4': 'https://www.adrianbulat.com/downloads/python-fan/3DFAN4-4a694010b9.zip',
    'depth': 'https://www.adrianbulat.com/downloads/python-fan/depth-6c4283c0e0.zip',
}

models_urls = {
    '1.6': {
        '2DFAN-4': 'https://www.adrianbulat.com/downloads/python-fan/2DFAN4_1.6-c827573f02.zip',
        '3DFAN-4': 'https://www.adrianbulat.com/downloads/python-fan/3DFAN4_1.6-ec5cf40a1d.zip',
        'depth': 'https://www.adrianbulat.com/downloads/python-fan/depth_1.6-2aa3f18772.zip',
    },
    '1.5': {
        '2DFAN-4': 'https://www.adrianbulat.com/downloads/python-fan/2DFAN4_1.5-a60332318a.zip',
        '3DFAN-4': 'https://www.adrianbulat.com/downloads/python-fan/3DFAN4_1.5-176570af4d.zip',
        'depth': 'https://www.adrianbulat.com/downloads/python-fan/depth_1.5-bc10f98e39.zip',
    },
}


class FaceAlignment:
    def __init__(self, landmarks_type, network_size=NetworkSize.LARGE,
                 device='cuda', flip_input=False, face_detector='sfd', face_detector_kwargs=None, verbose=False):
        self.device = device
        self.flip_input = flip_input
        self.landmarks_type = landmarks_type
        self.verbose = verbose

        if LooseVersion(torch.__version__) < LooseVersion('1.5.0'):
            raise ImportError(f'Unsupported pytorch version detected. Minimum supported version of pytorch: 1.5.0\
                            Either upgrade (recommended) your pytorch setup, or downgrade to face-alignment 1.2.0')

        network_size = int(network_size)
        pytorch_version = torch.__version__
        if 'dev' in pytorch_version:
            pytorch_version = pytorch_version.rsplit('.', 2)[0]
        else:
            pytorch_version = pytorch_version.rsplit('.', 1)[0]

        if 'cuda' in device:
            torch.backends.cudnn.benchmark = True

        # Get the face detector
        face_detector_module = __import__('face_alignment.detection.' + face_detector, globals(), locals(),
                                          [face_detector], 0)
        face_detector_kwargs = face_detector_kwargs or {}
        self.face_detector = face_detector_module.FaceDetector(device=device, verbose=verbose, **face_detector_kwargs)

        # Initialise the face alignemnt networks
        if landmarks_type == LandmarksType._2D:
            network_name = '2DFAN-' + str(network_size)
        else:
            network_name = '3DFAN-' + str(network_size)
        self.face_alignment_net = torch.jit.load(
            load_file_from_url(models_urls.get(pytorch_version, default_model_urls)[network_name]))

        self.face_alignment_net.eval()
        for param in self.face_alignment_net.parameters():
            param.requires_grad = False
        self.face_alignment_net.to(device)

    def get_landmarks_from_tensor(self, image):
        inp = image.to(self.device)
        inp = inp.transpose(2, 0) / 255.0
        inp = torch.unsqueeze(inp, dim=0)
        out = self.face_alignment_net(inp)

        return out

    def get_landmarks_from_numpy(self, image):
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        detected_faces = self.face_detector.detect_from_image(image)

        if len(detected_faces) == 0:
            warnings.warn("No faces were detected.")
            return None

        d = detected_faces[0]
        center = torch.tensor(
            [d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0])
        center[1] = center[1] - (d[3] - d[1]) * 0.12
        scale = (d[2] - d[0] + d[3] - d[1]) / self.face_detector.reference_scale
        inp = self.crop(image, center.cpu().numpy(), scale)  # crop to 256*256
        inp = torch.from_numpy(inp.transpose((2, 0, 1)))
        inp = torch.unsqueeze(inp, dim=0) / 255.0
        inp = inp.to(self.device)
        out = self.face_alignment_net(inp)

        return out

    def to8b(self, x):
        return (255 * np.clip(x, 0, 1)).astype(np.uint8)

    def crop(self, image, center, scale, resolution=256.0):
        ul = transform([1, 1], center, scale, resolution, True).cpu().numpy()
        br = transform([resolution, resolution], center, scale, resolution, True).cpu().numpy()
        # pad = math.ceil(torch.norm((ul - br).float()) / 2.0 - (br[0] - ul[0]) / 2.0)
        if image.ndim > 2:
            newDim = np.array([br[1] - ul[1], br[0] - ul[0],
                               image.shape[2]], dtype=np.int32)
            newImg = np.zeros(newDim, dtype=np.uint8)
        else:
            newDim = np.array([br[1] - ul[1], br[0] - ul[0]], dtype=np.int)
            newImg = np.zeros(newDim, dtype=np.uint8)
        ht = image.shape[0]
        wd = image.shape[1]
        newX = np.array(
            [max(1, -ul[0] + 1), min(br[0], wd) - ul[0]], dtype=np.int32)
        newY = np.array(
            [max(1, -ul[1] + 1), min(br[1], ht) - ul[1]], dtype=np.int32)
        oldX = np.array([max(1, ul[0] + 1), min(br[0], wd)], dtype=np.int32)
        oldY = np.array([max(1, ul[1] + 1), min(br[1], ht)], dtype=np.int32)
        newImg[newY[0] - 1:newY[1], newX[0] - 1:newX[1]
        ] = image[oldY[0] - 1:oldY[1], oldX[0] - 1:oldX[1], :]
        newImg = cv2.resize(newImg, dsize=(int(resolution), int(resolution)),
                            interpolation=cv2.INTER_LINEAR)
        return newImg


class LandmarkLoss(nn.Module):
    def __init__(self, device):
        super(LandmarkLoss, self).__init__()
        self.model = FaceAlignment(LandmarksType._2D, device=device)
        self.model.face_alignment_net.eval()
        self.default_loss = None  # 保存loss的历史最大值

    def forward(self, fake, target):
        fake_features = self.model.get_landmarks_from_tensor(fake)
        if fake_features is None:
            return None

        target_features = self.model.get_landmarks_from_numpy(target)
        landmark_loss = torch.zeros(1)
        for fake_feat, target_feat in zip(fake_features, target_features):
            landmark_loss += F.l1_loss(fake_feat, target_feat)

        total_loss = torch.sum(landmark_loss)

        return total_loss


if __name__ == '__main__':
    fa = FaceAlignment(LandmarksType._2D, device='cpu')
    fake = imageio.imread('output/test_input/24_rgb.jpg')
    fake = torch.tensor(fake, dtype=torch.float32, requires_grad=True)
    target = imageio.imread('dataset/Chinese/ori_imgs/0.jpg')

    landmarkLoss = LandmarkLoss(device='cuda' if torch.cuda.is_available() else 'cpu')
    print(landmarkLoss(fake, target))
