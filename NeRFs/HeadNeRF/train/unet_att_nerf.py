import os
import sys
import numpy as np
from torch import nn

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = f'{cur_path}/../../..'
sys.path.append(root_path)

from NeRFs.HeadNeRF.helper import *
import logging
from torch.utils.data import DataLoader
from utils.load_data.get_data import GetData
import torch.nn.functional  as F
import torch.optim
from natsort import natsorted
from tqdm import tqdm, trange

from models.attsets import AttentionSets
from models.audio_net import AudioNet, AudioAttNet
from models.face_nerf import FaceNeRF
from models.face_unet import FaceUNetCNN
from models.nerf_attention_model import NeRFAttentionModel
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
                                         input_ch=input_ch, dim_aud=args.dim_aud,
                                         output_ch=self.output_ch, skips=self.skips,
                                         attention_cnn_features=512,
                                         input_ch_views=input_ch_views,
                                         use_viewdirs=args.use_viewdirs)
        self.face_nerf_fine = FaceNeRF(D=args.netdepth, W=args.netwidth,
                                       input_ch=input_ch, dim_aud=args.dim_aud,
                                       output_ch=self.output_ch, skips=self.skips, attention_cnn_features=512,
                                       input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs)
        self.attention_block = AttentionSets(input_ch=(2 * attention_embed_out_dim + 128 + 2) + input_ch,
                                             attention_output_length=512)
        self.nerf_att_coarse = NeRFAttentionModel(self.face_nerf_coarse, self.attention_block, input_ch)
        self.nerf_att_fine = NeRFAttentionModel(self.face_nerf_fine, self.attention_block, input_ch)
        self.face_unet = FaceUNetCNN(embed_ln=attention_embed_out_dim * 2, input_ch=attention_embed_out_dim * 2)
        self.aud_net = AudioNet(args.dim_aud, args.win_size)
        self.aud_att_net = AudioAttNet()

    def forward(self, inputs):
        x, global_step, dataset_size = inputs
        batch_rays, target_s, bg_img, auds, raw_img, pose, landmark, index = x
        batch_rays, target_s, bg_img, auds, raw_img, pose, landmark = \
            torch.squeeze(batch_rays), torch.squeeze(target_s), torch.squeeze(bg_img), torch.squeeze(
                auds), torch.squeeze(raw_img), torch.squeeze(pose), torch.squeeze(landmark)

        batch_rays, target_s, bg_img, auds, raw_img, pose, landmark = batch_rays.type(
            torch.FloatTensor).cuda(), target_s.type(
            torch.FloatTensor).cuda(), bg_img.type(torch.FloatTensor).cuda(), auds.type(
            torch.FloatTensor).cuda(), raw_img.type(
            torch.FloatTensor).cuda(), pose.type(torch.FloatTensor).cuda(), landmark.type(torch.FloatTensor).cuda()

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

        return self.render_dynamic_face(self.H, self.W, self.focal, images=raw_img,
                                        lms_features=landmark,
                                        poses=pose, intrinsic=self.intrinsic, render_poses=render_poses,
                                        chunk=args.chunk,
                                        near=self.near, far=self.far, rays=batch_rays, bc_rgb=bg_img,
                                        aud_para=aud_feature)

    def render_dynamic_face(self, H, W, focal, images, lms_features, poses, intrinsic, render_poses=None,  # GRF 新增参数
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

        # 使用UNet对图像特征进行提取
        view_points = poses[..., 3]
        embeded_view_points = attention_embed_func(view_points)
        bc_viewpoints = einops.repeat(embeded_view_points, "w -> w h c", h=images.shape[0], c=images.shape[1])

        bc_viewpoints = torch.transpose(bc_viewpoints, 0, 2)
        rgb_vp = torch.cat([attention_embed_func(images), bc_viewpoints], -1)
        images_features = self.face_unet(rgb_vp[None, ...])

        # Render and reshape
        all_ret = self.batchify_rays(rays, bc_rgb, aud_para, poses=poses, intrinsic=intrinsic,
                                     images_features=images_features, lms_features=lms_features,
                                     chunk=chunk)
        for k in all_ret:
            k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
            all_ret[k] = torch.reshape(all_ret[k], k_sh)

        k_extract = ['rgb_map', 'disp_map', 'acc_map', 'last_weight']
        ret_list = [all_ret[k] for k in k_extract]
        ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
        return ret_list + [ret_dict]

    def batchify_rays(self, rays, bc_rgb, aud_para, poses, intrinsic, images_features, lms_features,
                      chunk=1024 * 32):
        """
            Render rays in smaller minibatches to avoid OOM.
        """
        all_ret = {}
        for i in range(0, rays.shape[0], chunk):
            ret = self.render_rays(rays[i:i + chunk], bc_rgb[i:i + chunk], aud_para,
                                   poses, intrinsic, images_features, lms_features)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])

        all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
        return all_ret

    def render_rays(self, rays, bc_rgb, aud_para, poses, intrinsic, images_features, lms_features,
                    retraw=False, lindisp=False, perturb=args.perturb, white_bkgd=False,
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
        raw = self.run_network(pts, lms_features, viewdirs, aud_para, self.nerf_att_coarse, intrinsic,
                               images_features,
                               poses, netchunk=args.netchunk)

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

            raw = self.run_network(pts, lms_features, viewdirs, aud_para, self.nerf_att_fine, intrinsic,
                                   images_features, poses, netchunk=args.netchunk)

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
        def raw2alpha(raw, dists, act_fn=F.relu):
            return 1. - torch.exp(-(act_fn(raw) + 1e-6) * dists)

        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape).cuda()],
                          -1)  # [N_rays, N_samples]

        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
        rgb = torch.cat((rgb[:, :-1, :], bc_rgb.unsqueeze(1)), dim=1)
        noise = 0.
        if raw_noise_std > 0.:
            noise = torch.randn(raw[..., 3].shape) * raw_noise_std

            # Overwrite randomly sampled data if pytest
            if pytest:
                np.random.seed(0)
                noise = np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std
                noise = torch.Tensor(noise).cuda()

        alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
        # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
        weights = alpha * \
                  torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).cuda(), 1. - alpha + 1e-10], -1),
                                -1).cuda()[:,
                  :-1]
        rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

        depth_map = torch.sum(weights * z_vals, -1)
        disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map).cuda(), depth_map / torch.sum(weights, -1))
        acc_map = torch.sum(weights, -1)

        if white_bkgd:
            rgb_map = rgb_map + (1. - acc_map[..., None])

        return rgb_map, disp_map, acc_map, weights, depth_map

    def run_network(self, inputs, lms_features, viewdirs, aud_para, nerf_model, intrisic, cnn_features, attention_poses,
                    netchunk=1024 * 64):
        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])  # (N_rays * N_samples, 3)
        embeded = embed_fn(inputs_flat)  # (N_rays * N_samples, 63)
        aud = aud_para.unsqueeze(0).expand(inputs_flat.shape[0], -1)  # (N_rays * N_samples, 64)
        # TODO: 怎么利用lms特征
        # lms_features = einops.repeat(lms_features, "h -> c h", c=inputs_flat.shape[0])
        # embed直接concat了音频特征
        embeded = torch.cat((embeded, aud), -1)  # (N_rays * N_samples, 64 + 63)
        if viewdirs is not None:
            input_dirs = viewdirs[:, None].expand(inputs.shape)
            input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
            embeded_dirs = embed_dirs_fn(input_dirs_flat)
            embeded = torch.cat([embeded, embeded_dirs], -1)  # (N_rays * N_samples, 64 + 63 + 27)

        # embeded = torch.cat((embeded, lms_features), -1) # TODO: 想想怎么使用lms特征
        outputs_flat = None
        for i in range(0, embeded.shape[0], netchunk):
            output = nerf_model([embeded[i:i + netchunk],
                                 gather_indices(inputs_flat[i:i + netchunk], attention_poses, intrisic,
                                                cnn_features)])
            if outputs_flat is None:
                outputs_flat = output
            else:
                outputs_flat = torch.cat([outputs_flat, output], dim=0)

        outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
        return outputs


def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d) \
            or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
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

    dataset_train = GetData(args.datadir, args.aud_file, mode="train", args=args)
    train_loader = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_work)

    dataset_val = GetData(args.datadir, args.aud_file, mode="val", args=args, skip=args.testskip)
    val_loader = DataLoader(dataset=dataset_val, batch_size=args.batch_size, shuffle=True, num_workers=args.num_work)

    write_config(args)

    # Create nerf model
    network = Network(H, W, focal, near=args.near, far=args.far, chunk=args.chunk, intrinsic=intrinsic,
                      N_samlpes=args.N_samples, N_importance=args.N_importance)

    # Load checkpoints
    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in natsorted(
            os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    logger.info(f'Found ckpts:{ckpts}')

    global_step = 0
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        logger.info(f'Reloading from: {ckpt_path}')
        ckpt = torch.load(ckpt_path)

        global_step = ckpt['global_step']

        network.face_nerf_coarse.load_state_dict(ckpt['network_fn_state_dict'])
        network.face_nerf_fine.load_state_dict(ckpt['network_fine_state_dict'])
        network.aud_net.load_state_dict(ckpt['network_audnet_state_dict'])
        network.aud_att_net.load_state_dict(ckpt['network_audattnet_state_dict'])

        # network.load_state_dict(ckpt['model_state_dict'])
        # optimizer.load_state_dict(ckpt['optimizer'])
        # scheduler.load_state_dict(ckpt['scheduler'])

    network = nn.DataParallel(network)
    network.apply(init_weights)
    optimizer = torch.optim.Adam(params=list(network.parameters()), lr=args.lrate, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    start = int(global_step / dataset_train.data_size)
    logger.info(f"start: {start}, global_step:{global_step}")
    for epoch in trange(start, N_iters):
        for iter, data in enumerate(tqdm(train_loader)):

            batch_rays, target_s, bg_img, auds, raw_img, pose, landmark, index = data
            rgb, disp, acc, _, extras = network([data, global_step, dataset_train.data_size])

            target_s = einops.rearrange(target_s, "h w c -> (h w) c")

            optimizer.zero_grad()

            # 前68个点是关键点的位置
            img_loss = img2mse(rgb, target_s)

            mouth_mse_loss = img2mse(rgb[48:68], target_s[48:68])

            loss = img_loss + mouth_mse_loss * 0
            psnr = mse2psnr(img_loss)

            if 'rgb0' in extras:
                img_loss0 = img2mse(extras['rgb0'], target_s)
                mouth_mse_loss0 = img2mse(extras['rgb0'][48:68], target_s[48:68])
                loss = loss + img_loss0 + mouth_mse_loss0 * 0

            loss.backward()

            optimizer.step()

            if global_step % args.i_print == 0:
                logger.info(
                    f"[TRAIN] epoch: {epoch} Iter: {iter} Loss: {loss.item()}  PSNR: {psnr.item()} LR: {scheduler.get_last_lr()[0]}")
                torch_writer.add_scalar('loss', loss.item(), global_step=global_step)
                torch_writer.add_scalar('psnr', psnr.item(), global_step=global_step)
                torch_writer.add_scalar('learning_rate', scheduler.get_last_lr()[0], global_step=global_step)

            # Update Learning Rate
            global_step += 1
            decay_rate = 0.1
            decay_steps = args.lrate_decay * 1500
            new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate

            # Rest is logging
            if global_step % args.i_weights == 0:
                path = os.path.join(basedir, expname, '{:06d}_head.tar'.format(epoch))
                torch.save({
                    'global_step': global_step,
                    'model_state_dict': network.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }, path)
                logger.info(f'Saved checkpoints at {path} and start to test with network')

                # 这里如果不测试的话可以加快一些进度
                network.eval()
                for val_i, data in enumerate(tqdm(val_loader)):
                    with torch.no_grad():
                        rgb, disp, acc, last_weight, _ = network([data, global_step, dataset_val.data_size])
                        rgb = einops.rearrange(rgb, 'h w c -> c h w')
                        rgb_cpu = to8b(rgb.cpu().numpy())
                        torch_writer.add_image("render_images", rgb_cpu, global_step=global_step)
                network.train()
                logger.info('Saved test set and turn back to training mode')

        # scheduler.step()


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()
