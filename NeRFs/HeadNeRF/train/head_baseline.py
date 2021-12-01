import json
import os
import sys
from multiprocessing import set_start_method

import einops
import imageio

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = f'{cur_path}/../../..'
sys.path.append(root_path)

from NeRFs.HeadNeRF.helper import *
from NeRFs.HeadNeRF.train.baseline import raw2outputs
from models.audio_net import AudioNet, AudioAttNet
import logging
from torch.utils.data import DataLoader, Dataset
import torch.optim
from natsort import natsorted
from tqdm import tqdm, trange

from models.face_nerf import FaceNeRF
import gc
from torch.utils.tensorboard import SummaryWriter

parser = config_parser()
args = parser.parse_args()

np.random.seed(0)

logger = logging.getLogger('adnerf')
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)

torch_writer = SummaryWriter(args.vis_path)
torch.set_default_tensor_type('torch.cuda.FloatTensor')
embed_fn, input_ch = get_embedder(args.multires, args.i_embed)  # input_ch = 63
embed_dirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
attention_embed_func, attention_embed_out_dim = get_embedder(5, 0)  # embed_out_dim = 33

try:
    set_start_method('spawn')
except RuntimeError:
    pass


class GetData(Dataset):
    # mode is [train, val, test]
    def __init__(self, data_dir, aud_file, mode, args, skip=1):
        self.data_dir = data_dir
        self.aud_file = aud_file
        self.mode = mode
        self.meta = None
        with open(os.path.join(data_dir, 'transforms_exp_{}.json'.format(mode)), 'r') as fp:
            self.meta = json.load(fp)
        self.all_imgs = []
        self.all_poses = []
        self.auds = []
        self.all_face_rects = []
        self.aud_features = np.load(os.path.join(self.data_dir, aud_file))
        self.background_img = torch.tensor(imageio.imread(os.path.join(data_dir, 'bc.jpg')) / 255.0)
        self.all_landmarks = []

        self.skip = 1 if self.mode == "train" else args.testskip

        for frame in self.meta['frames'][::skip]:
            fname = os.path.join(self.data_dir, 'head_imgs', str(frame['img_id']) + '.jpg')
            landmark = os.path.join(self.data_dir, 'ori_imgs', str(frame['img_id']) + '.lms')

            self.all_landmarks.append(landmark)
            self.all_imgs.append(fname)
            self.all_poses.append(np.array(frame['transform_matrix']))
            self.auds.append(self.aud_features[min(frame['aud_id'], self.aud_features.shape[0] - 1)])
            self.all_face_rects.append(np.array(frame['face_rect'], dtype=np.int32))  # 人脸的Bbox

        self.data_size = len(self.all_imgs)
        self.focal, self.cx, self.cy = float(self.meta['focal_len']), float(self.meta['cx']), float(self.meta['cy'])
        self.H, self.W = int(self.cy * 2), int(self.cx * 2)
        self.args = args

    def __getitem__(self, index):
        """ 支持下标索引，通过index把dataset中的数据拿出来"""
        if index is None:
            """ 未传下标时使用随机的方法采样 """
            index = np.random.choice(self.data_size)

        # 目标图像
        raw_img = torch.tensor(imageio.imread(self.all_imgs[index]))
        self.H, self.W = raw_img.shape[0], raw_img.shape[1]
        # 这里target只是用来比较
        target = torch.as_tensor(raw_img).cuda().float() / 255.0
        pose = self.all_poses[index][:3, :4]
        rect = self.all_face_rects[index]  # 人脸的方框
        landmark = np.loadtxt(self.all_landmarks[index])  # (68 * 2)
        batch_rays, target_s, bc_rgb = self.sample_rays(pose, rect, target, self.background_img, landmark)

        # bc_rgb是通过射线采样得到的
        bc_rgb = bc_rgb if self.mode == 'train' else self.background_img

        return batch_rays, target_s, bc_rgb, torch.tensor(self.auds, dtype=torch.float), \
               raw_img, pose, index

    def __len__(self):
        """返回数据集条数"""
        return self.data_size

    def get_rays(self, H, W, focal, c2w, cx=None, cy=None):
        i, j = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H))
        i = i.t()
        j = j.t()
        if cx is None:
            cx = W * .5
        if cy is None:
            cy = H * .5
        dirs = torch.stack(
            [(i - cx) / focal, -(j - cy) / focal, -torch.ones_like(i)], -1).to(c2w.device)
        # Rotate ray directions from camera frame to the world frame
        # dot product, equals to: [c2w.dot(dir) for dir in dirs]
        rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
        # Translate camera frame's origin to the world frame. It is the origin of all rays.
        rays_o = c2w[:3, -1].expand(rays_d.shape)
        return rays_o, rays_d

    def sample_rays(self, pose, face_rect, target, bc_img, landmark):
        rays_o, rays_d = self.get_rays(self.H, self.W, self.focal, torch.Tensor(pose).cuda(), self.cx,
                                       self.cy)  # (H, W, 3), (H, W, 3)
        landmark_mouth = landmark[48:]
        max_x, min_x = np.max(landmark_mouth[:, 0]) + 20, np.min(landmark_mouth[:, 0]) - 20
        max_y, min_y = np.max(landmark_mouth[:, 1]) + 20, np.min(landmark_mouth[:, 1]) - 20

        coords = torch.stack(torch.meshgrid(torch.linspace(0, self.H - 1, self.H),
                                            torch.linspace(0, self.W - 1, self.W)), -1)  # (H, W, 2)
        coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)

        # 这里是计算人脸的位置
        mouth_Winds = (coords[:, 0] >= min_x) & (coords[:, 0] <= max_x) & \
                      (coords[:, 1] >= min_y) & (coords[:, 1] <= max_y)

        rect_Winds = (coords[:, 0] >= face_rect[0]) & (coords[:, 0] <= face_rect[0] + face_rect[2]) & \
                     (coords[:, 1] >= face_rect[1]) & (coords[:, 1] <= face_rect[1] + face_rect[3])

        coords_mouth = coords[mouth_Winds]  # 人脸嘴部
        coords_rect = coords[rect_Winds & ~mouth_Winds]  # 包含人脸且不包含嘴部
        coords_norect = coords[~rect_Winds]  # 人脸之外的部分

        """ 计算除了关键点之外的采样数量 """
        sample_num = self.args.N_rand
        mouth_num = 0
        rect_num = int(sample_num * self.args.sample_rate)
        norect_num = sample_num - rect_num

        select_inds_rect = np.random.choice(coords_rect.shape[0], size=[rect_num], replace=False)
        select_coords_rect = coords_rect[select_inds_rect].long()

        select_inds_norect = np.random.choice(coords_norect.shape[0], size=[norect_num], replace=False)
        select_coords_norect = coords_norect[select_inds_norect].long()

        select_inds_mouth = np.random.choice(coords_mouth.shape[0], size=[mouth_num], replace=False)
        select_coords_mouth = coords_norect[select_inds_mouth].long()
        select_coords = torch.cat((select_coords_rect, select_coords_norect, select_coords_mouth), dim=0).long()

        rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        batch_rays = torch.stack([rays_o, rays_d], 0)
        target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        bc_rgb = bc_img[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

        return batch_rays, target_s, bc_rgb


class Network(nn.Module):
    def __init__(self, H, W, focal, near, far, chunk, intrinsic, N_samlpes, N_importance):
        super(Network, self).__init__()
        self.H = H
        self.W = W
        self.focal = focal
        self.near = near
        self.far = far
        self.chunk = chunk
        self.intrinsic = intrinsic
        self.N_samples = N_samlpes
        self.N_importance = N_importance
        self.output_ch = 4
        self.skips = [4]

        self.face_nerf_coarse = FaceNeRF(D=args.netdepth, W=args.netwidth,
                                         input_ch=input_ch, dim_aud=args.dim_aud, dim_latent=0, dim_expr=0,
                                         output_ch=self.output_ch, skips=self.skips,
                                         input_ch_views=input_ch_views,
                                         use_viewdirs=args.use_viewdirs)
        self.face_nerf_fine = FaceNeRF(D=args.netdepth, W=args.netwidth,
                                       input_ch=input_ch, dim_aud=args.dim_aud, dim_latent=0, dim_expr=0,
                                       output_ch=self.output_ch, skips=self.skips, input_ch_views=input_ch_views,
                                       use_viewdirs=args.use_viewdirs)
        self.aud_net = AudioNet(args.dim_aud, args.win_size)
        self.aud_att_net = AudioAttNet()

    def forward(self, inputs):
        x, global_step, dataset_size = inputs
        batch_rays, target_s, bg_img, auds, raw_img, pose, index = x
        batch_rays, target_s, bg_img, auds, raw_img, pose, = \
            torch.squeeze(batch_rays), torch.squeeze(target_s), torch.squeeze(bg_img), torch.squeeze(
                auds), torch.squeeze(raw_img), torch.squeeze(pose),

        batch_rays, target_s, bg_img, auds, raw_img, pose = batch_rays.type(
            torch.FloatTensor).cuda(), target_s.type(
            torch.FloatTensor).cuda(), bg_img.type(torch.FloatTensor).cuda(), auds.type(
            torch.FloatTensor).cuda(), raw_img.type(
            torch.FloatTensor).cuda(), pose.type(torch.FloatTensor).cuda()

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
            aud_feature = self.aud_net(auds[index])

        render_poses = None if self.training is True else pose[:3, :4]

        return self.render_dynamic_face(raw_img.shape[0], raw_img.shape[1], self.focal,
                                        poses=pose, intrinsic=self.intrinsic, render_poses=render_poses,
                                        chunk=args.chunk, near=self.near, far=self.far, rays=batch_rays, bc_rgb=bg_img,
                                        aud_para=aud_feature, ndc=False)

    def batchify_rays(self, rays, bc_rgb, aud_para, poses, intrinsic, images_features, chunk=1024 * 32):
        """
            Render rays in smaller minibatches to avoid OOM. -> 异步地加入GPU不能防止OOM，实现不了想要的效果
        """
        all_ret = {}
        for i in range(0, rays.shape[0], chunk):
            ret = self.render_rays(rays[i:i + chunk], bc_rgb[i:i + chunk], aud_para)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])

        all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
        return all_ret

    def render_rays(self, rays, bc_rgb, aud_para, retraw=False, lindisp=False, perturb=args.perturb, white_bkgd=False,
                    raw_noise_std=0., attention_embed_ln=0, pytest=False):
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

            rgb_map, disp_map, acc_map, weights, depth_map = self.raw2outputs(
                raw, z_vals, rays_d, bc_rgb, raw_noise_std, white_bkgd, pytest=pytest)

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
        if viewdirs is not None:
            input_dirs = viewdirs[:, None].expand(inputs.shape)
            input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
            embeded_dirs = embed_dirs_fn(input_dirs_flat)
            embeded = torch.cat([embeded, embeded_dirs], -1)  # (N_rays * N_samples, 64 + 63 + 27)

        outputs_flat = None
        for i in range(0, embeded.shape[0], netchunk):
            output = nerf_model(embeded[i:i + netchunk], aud_para)
            if outputs_flat is None:
                outputs_flat = output
            else:
                outputs_flat = torch.cat([outputs_flat, output], dim=0)

        outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
        return outputs

    def render_dynamic_face(self, H, W, focal, poses, intrinsic, render_poses=None,  # GRF 新增参数
                            chunk=1024 * 32, near=0., far=1.,
                            rays=None, bc_rgb=None, aud_para=None, ndc=False, use_viewdirs=args.use_viewdirs):
        if render_poses is not None:
            rays_o, rays_d = get_rays(H, W, focal, render_poses)
            bc_rgb = einops.rearrange(bc_rgb, 'h w c -> (h w) c')
        else:
            rays_o, rays_d = rays

        if use_viewdirs:  # True
            # provide ray directions as input
            viewdirs = rays_d
            viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
            viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

        sh = rays_d.shape  # [..., 3]
        if ndc:
            # for forward facing scenes
            rays_o, rays_d = ndc_rays(H, W, focal, 1., rays_o, rays_d)

        # Create ray batch
        rays_o = torch.reshape(rays_o, [-1, 3]).float()
        rays_d = torch.reshape(rays_d, [-1, 3]).float()

        near = near * torch.ones_like(rays_d[..., :1])
        far = far * torch.ones_like(rays_d[..., :1])

        rays = torch.cat([rays_o, rays_d, near, far], -1)
        if use_viewdirs:
            rays = torch.cat([rays, viewdirs], -1)

        # Render and reshape
        all_ret = self.batchify_rays(rays, bc_rgb, aud_para, poses=poses, intrinsic=intrinsic,
                                     images_features=None, chunk=chunk)
        for k in all_ret:
            k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
            all_ret[k] = torch.reshape(all_ret[k], k_sh)

        k_extract = ['rgb_map', 'disp_map', 'acc_map', 'last_weight']
        ret_list = [all_ret[k] for k in k_extract]
        ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
        return ret_list + [ret_dict]


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    if isinstance(m, nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def train():
    # Def: 超参
    H, W, focal, cx, cy, = 450, 450, 1200.0, 225.0, 225.0
    intrinsic = np.array([[focal, 0., W / 2], [0, focal, H / 2], [0, 0, 1.]])
    hwfcxy = [H, W, focal, cx, cy]
    basedir, expname = args.basedir, args.expname

    N_rand, use_batching = args.N_rand, args.use_batching
    logger.info(
        f'N_rand: {N_rand} , batch_size: {args.batch_size}, sample_rate {args.sample_rate}, num_worker: {args.num_work}')

    N_iters = args.N_iters + 1
    logger.info('Begin')

    # 定义数据加载器
    dataset_train = GetData(args.datadir, args.aud_file, mode="train", args=args)
    train_loader = DataLoader(dataset=dataset_train, batch_size=args.batch_size, num_workers=args.num_work,
                              shuffle=False)
    dataset_val = GetData(args.datadir, args.aud_file, mode="val", args=args, skip=args.testskip)
    val_loader = DataLoader(dataset=dataset_val, batch_size=args.batch_size, shuffle=True, num_workers=args.num_work,
                            generator=torch.Generator(device='cuda'))
    # 写入config文件
    write_config(args)

    # 定义网络模型
    network = Network(H, W, focal, near=args.near, far=args.far, chunk=args.chunk, intrinsic=intrinsic,
                      N_samlpes=args.N_samples, N_importance=args.N_importance)

    global_step = 0
    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
        logger.info(f'Reloading from: {ckpts}')
        ckpt = torch.load(ckpts[-1])
        network.face_nerf_coarse.load_state_dict(ckpt['network_fn_state_dict'])
        network.face_nerf_fine.load_state_dict(ckpt['network_fine_state_dict'])
        network.aud_net.load_state_dict(ckpt['network_audnet_state_dict'])
        network.aud_att_net.load_state_dict(ckpt['network_audattnet_state_dict'])
    else:
        network = nn.DataParallel(network)
        network.apply(init_weights)
        ckpts = [os.path.join(basedir, expname, f) for f in natsorted(
            os.listdir(os.path.join(basedir, expname))) if '.tar' in f]
        if len(ckpts) > 0:
            ckpt_path = ckpts[-1]
            logger.info(f'Found ckpts:{ckpts}')
            ckpt = torch.load(ckpt_path)
            network.load_state_dict(ckpt['model_state_dict'])
            global_step = ckpt['global_step']

    # 定义优化器和学习率变化器
    optimizer = torch.optim.Adam(params=list(network.parameters()), lr=args.lrate, betas=(0.9, 0.999))

    start = int(global_step / dataset_train.data_size)
    logger.info(f"start: {start}, global_step:{global_step}")

    for epoch in trange(start, N_iters):
        for iter, data in enumerate(tqdm(train_loader)):
            batch_rays, target_s, bg_img, auds, raw_img, pose, index = data
            rgb, disp, acc, _, extras = network([data, global_step, dataset_train.data_size])

            target_s = einops.rearrange(target_s, "h w c -> (h w) c")

            optimizer.zero_grad()

            # 前68个点是关键点的位置
            img_loss = img2mse(rgb, target_s)
            loss = img_loss
            psnr = mse2psnr(img_loss)

            if 'rgb0' in extras:
                img_loss0 = img2mse(extras['rgb0'], target_s)
                loss = loss + img_loss0

            loss.backward()

            optimizer.step()

            decay_rate = 0.1
            decay_steps = args.lrate_decay * 1500
            new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate

            if global_step % args.i_print == 0:
                logger.info(
                    f"[TRAIN] epoch: {epoch} Iter: {iter} Loss: {loss.item()}  PSNR: {psnr.item()} LR: {new_lrate}")
                torch_writer.add_scalar('loss', loss.item(), global_step=global_step)
                torch_writer.add_scalar('psnr', psnr.item(), global_step=global_step)
                torch_writer.add_scalar('learning_rate', new_lrate, global_step=global_step)

            if global_step % (100 * args.i_print) == 0:
                network.eval()
                for val_i, data in enumerate(tqdm(val_loader)):
                    batch_rays, target_s, bg_img, auds, raw_img, pose, index = data
                    with torch.no_grad():
                        rgb, _, _, weights, _ = network([data, global_step, dataset_val.data_size])
                        break
                rgb = einops.rearrange(rgb, 'h w c -> c h w')
                raw_img = einops.reduce(raw_img, 'b h w c ->c h w', 'max') / 255.0
                pred_with_label = torch.cat((rgb.cpu(), raw_img.cpu()), dim=1)
                torch_writer.add_image("val/rgb_fine", pred_with_label, global_step=global_step)
                network.train()
                logger.info('Saved test set and turn back to trainning mode')

            # Rest is logging
            if global_step % args.i_weights == 0:
                path = os.path.join(basedir, expname, '{:06d}_head.tar'.format(epoch))
                torch.save({
                    'global_step': global_step,
                    'model_state_dict': network.state_dict(),
                    'optimizer': optimizer.state_dict()}, path)
                logger.info(f'Saved checkpoints at {path} and start to test with network')
            global_step += 1


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()
