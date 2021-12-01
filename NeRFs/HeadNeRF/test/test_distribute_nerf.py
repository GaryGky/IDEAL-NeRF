import json
import os
import sys

import cv2

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = f'{cur_path}/../../..'
sys.path.append(root_path)

os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'

import face_recognition
import imageio
from torch.multiprocessing import set_start_method
from torch.utils.data import Dataset  # Dataset是个抽象类，只能用于继承
from loss.vgg_face_loss import VGGFaceLoss
from loss.vgg_loss import VGGLOSS
from NeRFs.HeadNeRF.helper import *
from NeRFs.HeadNeRF.train.baseline import raw2outputs
from models.audio_net import AudioNet, AudioAttNet
import logging
from torch.utils.data import DataLoader
from utils.load_data.get_data_second_stage import GetData
import torch.optim
from natsort import natsorted
from tqdm import tqdm, trange
from models.face_nerf import FaceNeRF
from torch.utils.tensorboard import SummaryWriter
from loss.landmark_loss import LandmarkLoss

parser = config_parser()
args = parser.parse_args()

np.random.seed(0)

logger = logging.getLogger('adnerf')
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)

torch.set_default_tensor_type('torch.cuda.FloatTensor')
embed_fn, input_ch = get_embedder(args.multires, args.i_embed)  # input_ch = 63
embed_dirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
attention_embed_func, attention_embed_out_dim = get_embedder(5, 0)  # embed_out_dim = 33


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    if isinstance(m, nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


try:
    set_start_method('spawn')
except RuntimeError:
    pass


# dataloader
class GetData(Dataset):
    # mode is [train, val, test]
    # 从Noah中需要获取：Landmark Audio Pose
    # 从Obama中需要获取：Background HeadImage
    def __init__(self, source_dir, data_dir, aud_file, mode, args, skip=1):
        self.face_size = 225

        self.data_dir = data_dir
        self.aud_file = aud_file
        self.mode = mode
        self.meta = None
        with open(os.path.join(data_dir, 'transforms_{}.json'.format(mode)), 'r') as fp:
            self.meta = json.load(fp)
        self.all_imgs = []
        self.all_poses = []
        self.auds = []
        # 使用Noah的音频
        self.aud_features = np.load(os.path.join(source_dir, aud_file))
        self.background_img = \
            torch.tensor(imageio.imread(os.path.join(source_dir, 'bc.jpg')) / 255, dtype=torch.float32)
        self.skip = 1 if self.mode == "train" else args.testskip

        for frame in self.meta['frames'][::skip]:
            if frame['img_id'] > len(self.aud_features):
                break

            # 获取Obama头部的图像和pose
            fname = os.path.join(self.data_dir, 'head_imgs', str(frame['img_id']) + '.jpg')
            pose = np.array(frame['transform_matrix'])
            self.all_imgs.append(fname)
            self.all_poses.append(pose)

            aud = self.aud_features[min(frame['aud_id'], self.aud_features.shape[0] - 1)]
            self.auds.append(aud)

        self.data_size = min(len(self.all_imgs), len(self.aud_features))
        self.focal, self.cx, self.cy = float(self.meta['focal_len']), float(self.meta['cx']), float(self.meta['cy'])
        self.args = args

    def __getitem__(self, index):
        """ 支持下标索引，通过index把dataset中的数据拿出来"""
        if index is None:
            """ 未传下标时使用随机的方法采样 """
            index = np.random.choice(self.data_size)
        pose = torch.from_numpy(self.all_poses[index]).cuda()
        rays_o, rays_d = self.get_rays(H=self.face_size, W=self.face_size, focal=self.focal, c2w=pose)
        viewdirs = rays_d
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()
        # Create ray batch
        rays_o = torch.reshape(rays_o, [-1, 3]).float()
        rays_d = torch.reshape(rays_d, [-1, 3]).float()

        near = self.args.near * torch.ones_like(rays_d[..., :1])
        far = self.args.far * torch.ones_like(rays_d[..., :1])

        rays = torch.cat([rays_o, rays_d, near, far], -1)
        rays = torch.cat([rays, viewdirs], -1)

        auds = torch.tensor(self.auds)
        gt_image, target_lf = torch.zeros(1), torch.zeros(1)
        bc_rgb = torch.reshape(self.background_img, [-1, 3])
        return rays, bc_rgb, auds, pose, gt_image, target_lf, index

    def __len__(self):
        """返回数据集条数"""
        return self.data_size

    def get_rays(self, H, W, focal, c2w):
        return get_rays(H, W, focal, c2w)


class Network(nn.Module):
    def __init__(self, H, W, focal, near, far, chunk, N_samlpes, N_importance):
        super(Network, self).__init__()
        self.H = H
        self.W = W
        self.focal = focal
        self.near = near
        self.far = far
        self.chunk = chunk
        self.N_samples = N_samlpes
        self.N_importance = N_importance
        self.output_ch = 4
        self.skips = [4]

        self.face_nerf_coarse = FaceNeRF(D=args.netdepth, W=args.netwidth,
                                         input_ch=input_ch, dim_aud=args.dim_aud,
                                         output_ch=self.output_ch, skips=self.skips,
                                         attention_cnn_features=0,
                                         input_ch_views=input_ch_views,
                                         use_viewdirs=args.use_viewdirs)
        self.face_nerf_fine = FaceNeRF(D=args.netdepth, W=args.netwidth,
                                       input_ch=input_ch, dim_aud=args.dim_aud,
                                       output_ch=self.output_ch, skips=self.skips, attention_cnn_features=0,
                                       input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs)

        self.aud_net = AudioNet(args.dim_aud, args.win_size)
        self.aud_att_net = AudioAttNet()

    def forward(self, inputs):
        x, global_step, dataset_size = inputs
        rays, bg_img, auds, pose, index = x

        # squeeze
        rays, bg_img, auds, pose = torch.squeeze(rays).type(torch.float).cuda(), \
                                   torch.squeeze(bg_img).type(torch.float).cuda(), \
                                   torch.squeeze(auds).type(torch.float), \
                                   torch.squeeze(pose).type(torch.float).cuda()

        if global_step >= args.nosmo_iters:
            if global_step == args.nosmo_iters:
                logger.info("+++++++ Change AudNet To AudAttNet +++++++")
            # args.smo_size=8
            smo_half_win = int(args.smo_size / 2)
            left_i = index - smo_half_win
            right_i = index + smo_half_win
            pad_left, pad_right = 0, 0
            if left_i < 0:
                pad_left = -left_i
                left_i = 0
            if right_i > dataset_size:
                pad_right = right_i - dataset_size
                right_i = dataset_size
            auds_win = auds[left_i:right_i]
            if pad_left > 0:
                auds_win = torch.cat(
                    (torch.zeros_like(auds_win)[:pad_left], auds_win), dim=0)
            if pad_right > 0:
                auds_win = torch.cat(
                    (auds_win, torch.zeros_like(auds_win)[:pad_right]), dim=0)
            auds_win = self.aud_net(auds_win)
            aud_feature = self.aud_att_net(auds_win)
        else:
            aud_feature = self.aud_net(auds[min(index, len(auds))])

        all_ret = self.batchify_rays(rays, bg_img, aud_feature, pose, args.chunk)
        return all_ret['rgb_map'], all_ret['rgb0']

    def batchify_rays(self, rays, bc_rgb, aud_para, poses, chunk=1024 * 32):
        """
            Render rays in smaller minibatches to avoid OOM.
        """
        all_ret = {}
        for i in range(0, rays.shape[0], chunk):
            ret = self.render_rays(rays[i:i + chunk], bc_rgb[i:i + chunk], aud_para, poses)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])

        all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
        return all_ret

    def render_rays(self, rays, bc_rgb, aud_para, poses, retraw=False, lindisp=False,
                    perturb=args.perturb, white_bkgd=False, raw_noise_std=0., attention_embed_ln=0, pytest=False):
        N_rays = rays.shape[0]
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # [N_rays, 3] each
        viewdirs = rays[:, -3:] if rays.shape[-1] > 8 else None
        bounds = torch.reshape(rays[..., 6:8], [-1, 1, 2])
        near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

        t_vals = torch.linspace(0., 1., steps=args.N_samples).cuda()
        if not lindisp:
            z_vals = near * (1. - t_vals) + far * (t_vals)
        else:
            z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

        z_vals = z_vals.expand([N_rays, args.N_samples])

        if perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).cuda()

            # Pytest, overwrite u with numpy's fixed random numbers
            if pytest:
                np.random.seed(0)
                t_rand = np.random.rand(*list(z_vals.shape))
                t_rand = torch.Tensor(t_rand).cuda()
            t_rand[..., -1] = 1.0
            z_vals = lower + (upper - lower) * t_rand
            z_vals.cuda()
            viewdirs.cuda()
        " N_rays条射线，每条射线上采样N_samples个点"
        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
              z_vals[..., :, None]  # [N_rays 2048, N_samples 64, 3]

        raw = self.run_network(pts, viewdirs, aud_para, self.face_nerf_coarse, netchunk=args.netchunk)

        rgb_map, disp_map, acc_map, weights, depth_map = self.raw2outputs(
            raw, z_vals, rays_d, bc_rgb, raw_noise_std, white_bkgd, pytest=pytest)

        if args.N_importance > 0:
            rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

            z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = sample_pdf(
                z_vals_mid, weights[..., 1:-1], args.N_importance, det=(perturb == 0.), pytest=pytest)
            z_samples = z_samples.detach()

            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            pts = rays_o[..., None, :] + rays_d[..., None, :] * \
                  z_vals[..., :, None]  # [N_rays, N_samples + N_importance, 3]

            raw = self.run_network(pts, viewdirs, aud_para, self.face_nerf_fine, netchunk=args.netchunk)

            rgb_map, disp_map, acc_map, weights, depth_map = self.raw2outputs(raw, z_vals, rays_d, bc_rgb,
                                                                              raw_noise_std, white_bkgd, pytest=pytest)

        ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map}
        if retraw:
            ret['raw'] = raw
        if args.N_importance > 0:
            ret['rgb0'] = rgb_map_0
            ret['disp0'] = disp_map_0
            ret['acc0'] = acc_map_0
            ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]
            ret['last_weight'] = weights[..., -1]

        for k in ret:
            if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()):
                logger.info(f"! [Numerical Error] {k} contains nan or inf.")

        return ret

    def raw2outputs(eslf, raw, z_vals, rays_d, bc_rgb, raw_noise_std=0, white_bkgd=False, pytest=False):
        return raw2outputs(raw, z_vals, rays_d, bc_rgb, raw_noise_std, white_bkgd, pytest)

    def run_network(self, inputs, viewdirs, aud_para, nerf_model, netchunk=1024 * 64):
        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])  # (N_rays * N_samples, 3)
        embeded = embed_fn(inputs_flat)  # (N_rays * N_samples, 63)
        aud = aud_para.unsqueeze(0).expand(inputs_flat.shape[0], -1)  # (N_rays * N_samples, 64)
        embeded = torch.cat((embeded, aud), -1)  # (N_rays * N_samples, 64 + 63)

        if viewdirs is not None:
            input_dirs = viewdirs[:, None].expand(inputs.shape)
            input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
            embeded_dirs = embed_dirs_fn(input_dirs_flat)
            embeded = torch.cat([embeded, embeded_dirs], -1)  # (N_rays * N_samples, 64 + 63 + 27)

        outputs_flat = None
        for i in range(0, embeded.shape[0], netchunk):
            output = nerf_model(embeded[i:i + netchunk])  # (n, 65536, 3)
            if outputs_flat is None:
                outputs_flat = output
            else:
                outputs_flat = torch.cat([outputs_flat, output], dim=0)

        outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
        return outputs


def train():
    # Def: 超参
    H, W, focal, cx, cy, = 450, 450, 1200.0, 225.0, 225.0
    intrinsic = np.array([[focal, 0., W / 2], [0, focal, H / 2], [0, 0, 1.]])
    hwfcxy = [H, W, focal, cx, cy]
    basedir, expname = args.basedir, args.expname

    N_rand, use_batching = args.N_rand, args.use_batching
    logger.info(
        f'N_rand: {N_rand} , N_samples: {args.N_samples}, N_importance {args.N_importance}, learning_rate: {args.lrate}')

    N_iters = args.N_iters + 1
    logger.info('Begin')

    # 定义数据加载器
    dataset_train = GetData(source_dir='dataset/Chinese', data_dir=args.datadir, aud_file=args.aud_file, mode="train",
                            args=args)
    train_loader = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_work)

    # 写入config文件
    write_config(args)

    # 加载已经保存的模型
    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in natsorted(
            os.listdir(os.path.join(basedir, expname))) if 'head.tar' in f]

    logger.info(f'Found ckpts:{ckpts}')

    # 定义网络模型
    network = Network(H, W, focal, near=args.near, far=args.far, chunk=args.chunk, N_samlpes=args.N_samples,
                      N_importance=args.N_importance)

    network = nn.DataParallel(network)

    network.apply(init_weights)

    global_step = 0
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        logger.info(f'Reloading from: {ckpt_path}')
        ckpt = torch.load(ckpt_path)
        network.load_state_dict(ckpt['model_state_dict'])

    start = int(global_step / dataset_train.data_size)
    logger.info(f"start: {start}, global_step:{global_step}, gpu_num:{args.gpu_num}")

    vid_out = cv2.VideoWriter(os.path.join('output/test_eval', 'Obama_ch.avi'),
                              cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 25,
                              (dataset_train.face_size, dataset_train.face_size))
    network.eval()
    with torch.no_grad():
        for epoch in trange(1):
            for iter, data in enumerate(tqdm(train_loader)):
                rays, bc_rgb, auds, pose, gt_img, landmark, index = data

                rays = einops.rearrange(rays, 'b (h w) c -> (b h) w c', h=args.gpu_num)
                bc_rgb = einops.rearrange(bc_rgb, 'b (h w) c -> (b h) w c', h=args.gpu_num)

                auds = einops.repeat(auds, 'b h w c -> (repeat b) h w c', repeat=args.gpu_num)
                pose = einops.repeat(pose, 'b h w -> (repeat b) h w', repeat=args.gpu_num)
                index = einops.repeat(index, 'h -> (repeat h)', repeat=args.gpu_num)

                inp = [rays, bc_rgb, auds, pose, index]

                rgb, rgb0 = network([inp, global_step, dataset_train.data_size])

                rgb_save = torch.reshape(rgb,
                                         [dataset_train.face_size, dataset_train.face_size, 3]).detach().cpu().numpy()
                vid_out.write(to8b(cv2.cvtColor(rgb_save, cv2.COLOR_BGR2RGB)))
            break
    vid_out.release()


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()
