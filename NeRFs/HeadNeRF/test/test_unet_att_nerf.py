import imageio
import torch
from mmcv import DataLoader
from natsort import natsorted
from tqdm import tqdm
import os, sys

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = f'{cur_path}/../../..'
sys.path.append(root_path)

from NeRFs.HeadNeRF.train.unet_att_nerf import Network
from NeRFs.HeadNeRF.helper import config_parser, to8b
from utils.load_data.get_data import GetData
from utils.log_utils import get_logger
import torch.nn as nn
import os
import numpy as np

logger = get_logger()


def test():
    parser = config_parser()
    args = parser.parse_args()
    basedir, expname = args.basedir, args.expname

    logger.info(f'datadir: {args.datadir}, aud_file:{args.aud_file}')
    logger.info(f'basedir: {basedir}, expname:{expname}')

    dataset_val = GetData(args.datadir, args.aud_file, mode="val", args=args, skip=args.testskip)
    val_loader = DataLoader(dataset=dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_work)

    H, W, focal = int(dataset_val.H), int(dataset_val.W), dataset_val.focal
    intrinsic = np.array([[focal, 0., W / 2], [0, focal, H / 2], [0, 0, 1.]])

    network = Network(H, W, focal, near=args.near, far=args.far, chunk=args.chunk, intrinsic=intrinsic,
                      N_samlpes=args.N_samples, N_importance=args.N_importance)
    network = nn.DataParallel(network)

    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in natsorted(
            os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    logger.info(f'Found ckpts:{ckpts}')

    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        logger.info(f'Reloading from: {ckpt_path}')
        ckpt = torch.load(ckpt_path)
        network.load_state_dict(ckpt['model_state_dict'])

    global_step = 0
    network.eval()

    for val_i, data in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            rgb, disp, acc, last_weight, _ = network([data, global_step, dataset_val.data_size])
            rgb_cpu = to8b(rgb.cpu().numpy())
            filename = f'output/ua_nerf/Noah_{val_i}.jpg'
            imageio.imwrite(filename, rgb_cpu)
            logger.info(f"render: {val_i} finished")


if __name__ == '__main__':
    test()
