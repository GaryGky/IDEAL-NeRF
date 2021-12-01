import json
import os
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = f'{cur_path}/../../..'
sys.path.append(root_path)

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

torch_writer = SummaryWriter(args.vis_path)
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
        self.face_size = 256

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
        self.aud_features = np.load(os.path.join(self.data_dir, aud_file))

        self.background_img = torch.tensor(imageio.imread(os.path.join(self.data_dir, 'bc.jpg')) / 255,
                                           dtype=torch.float32)
        self.all_landmarks = []
        self.skip = 1 if self.mode == "train" else args.testskip

        for frame in self.meta['frames'][::skip]:
            if frame['img_id'] > len(self.aud_features):
                break

            # 获取Obama头部的图像和pose
            fname = os.path.join(self.data_dir, 'head_imgs', str(frame['img_id']) + '.jpg')
            pose = np.array(frame['transform_matrix'])
            self.all_imgs.append(fname)
            self.all_poses.append(pose)

            # 获取Test的人脸图像和audio
            landmark = os.path.join(source_dir, 'ori_imgs', str(frame['img_id']) + '.jpg')
            if os.path.exists(landmark) is False:
                break

            aud = self.aud_features[min(frame['aud_id'], self.aud_features.shape[0] - 1)]
            self.auds.append(aud)
            self.all_landmarks.append(landmark)

        self.data_size = min(len(self.all_imgs), len(self.all_landmarks))
        self.focal, self.cx, self.cy = float(self.meta['focal_len']), float(self.meta['cx']), float(self.meta['cy'])
        self.args = args

    def __getitem__(self, index):
        """ 支持下标索引，通过index把dataset中的数据拿出来"""
        if index is None:
            """ 未传下标时使用随机的方法采样 """
            index = np.random.choice(self.data_size)

        # 目标图像
        raw_img = torch.tensor(imageio.imread(self.all_imgs[index]) / 255.0, dtype=torch.float32)
        self.H, self.W = raw_img.shape[0], raw_img.shape[1]
        # 这里target只是用来比较
        pose = torch.tensor(self.all_poses[index][:3, :4])

        # 第二阶段：直接采样整张图片会导致OOM，所以尝试采样Obama人脸部分
        image = face_recognition.load_image_file(self.all_imgs[index])
        face_rects = face_recognition.face_locations(image)
        face_rect = list(face_rects[0])

        auds = torch.tensor(self.auds, dtype=torch.float)  # audio 特征

        target_lf = imageio.imread(self.all_landmarks[index])  # Test人脸图像
        target_lf = torch.tensor(target_lf, dtype=torch.float32)

        # 裁剪出人脸的部分
        len_x, len_y = face_rect[2] - face_rect[0], face_rect[1] - face_rect[3]
        fill_x, fill_y = self.face_size - len_x, self.face_size - len_y
        left, top = fill_x // 2, fill_y // 2
        right, bottom = fill_x - left, fill_y - top

        face_rect[0], face_rect[2] = face_rect[0] - left, face_rect[2] + right
        face_rect[1], face_rect[3] = face_rect[1] + top, face_rect[3] - bottom

        # 采集射线
        [rays_o, rays_d], gt_image, bc_rgb = self.sample_rays(pose, face_rect, raw_img, self.background_img)
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

        # bc_rgb = einops.rearrange(bc_rgb, 'h w c -> (h w) c')
        return rays, bc_rgb, auds, pose, gt_image, target_lf, index

    def __len__(self):
        """返回数据集条数"""
        return self.data_size

    def sample_rays(self, pose, face_rect, target, bc_img):
        rays_o, rays_d = get_rays(self.H, self.W, self.focal, pose, self.cx, self.cy)

        coords = torch.stack(
            torch.meshgrid(torch.linspace(0, self.H - 1, self.H),
                           torch.linspace(0, self.W - 1, self.W)), -1)  # (H, W, 2)

        coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)

        # 这里是计算人脸的位置
        rect_inds = (coords[:, 0] >= face_rect[0]) & (coords[:, 0] <= face_rect[2]) \
                    & (coords[:, 1] >= face_rect[3]) & (coords[:, 1] <= face_rect[1])

        coords_rect = coords[rect_inds]  # 包含人脸的像素
        coords_norect = coords[~rect_inds]  # 不包含人脸的像素

        """ 计算除了关键点之外的采样数量 """
        sample_num = self.args.N_rand
        rect_num = int(sample_num * self.args.sample_rate)
        norect_num = sample_num - rect_num

        select_inds_rect = np.sort(np.random.choice(coords_rect.shape[0], size=[rect_num], replace=False))
        select_coords_rect = coords_rect[select_inds_rect].long()  # (N_rand * sample_rate, 2) 包含人脸像素的坐标

        select_inds_norect = np.sort(np.random.choice(coords_norect.shape[0], size=[norect_num], replace=False))
        select_coords_norect = coords_norect[select_inds_norect].long()  # (N_rand * (1-sample_rate), 2) 不包含人脸像素的坐标

        select_coords = torch.cat((select_coords_rect, select_coords_norect), dim=0).long()

        rays_o = torch.reshape(rays_o[face_rect[0]:face_rect[2], face_rect[3]:face_rect[1]], [-1, 3])  # (N_rand, 3)
        rays_d = torch.reshape(rays_d[face_rect[0]:face_rect[2], face_rect[3]:face_rect[1]], [-1, 3])  # (N_rand, 3)
        batch_rays = torch.stack([rays_o, rays_d], 0)

        target_s = target[face_rect[0]:face_rect[2], face_rect[3]:face_rect[1]]  # (N_rand, 3)
        bc_rgb = bc_img[face_rect[0]:face_rect[2], face_rect[3]:face_rect[1]]  # (N_rand, 3)

        target_s = torch.reshape(target_s, [-1, 3])
        bc_rgb = torch.reshape(bc_rgb, [-1, 3])
        return batch_rays, target_s, bc_rgb

    def get_rays(self, H, W, focal, c2w, rect):
        return get_rays(H, W, focal, c2w)


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
            aud_feature = self.aud_net(auds[min(index, len(auds) - 1)])

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

    logger.info(f'N_samples: {args.N_samples}, N_importance {args.N_importance}, learning_rate: {args.lrate}')

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
    network = Network(H, W, focal, near=args.near, far=args.far, chunk=args.chunk, intrinsic=intrinsic,
                      N_samlpes=args.N_samples, N_importance=args.N_importance)

    network = nn.DataParallel(network)

    network.apply(init_weights)

    # 定义优化器和学习率变化器
    optimizer = torch.optim.Adam(params=list(network.parameters()),
                                 lr=args.lrate,
                                 betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    vgg_loss = VGGLOSS()
    vgg_face_loss = VGGFaceLoss()
    landmark_loss = LandmarkLoss(device='cuda' if torch.cuda.is_available() else 'cpu')

    global_step = 0
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        logger.info(f'Reloading from: {ckpt_path}')
        ckpt = torch.load(ckpt_path)
        # global_step = ckpt['global_step']
        network.load_state_dict(ckpt['model_state_dict'])
        # optimizer.load_state_dict(ckpt['optimizer'])

        if 'scheduler' in ckpt:
            pass
            # scheduler.load_state_dict(ckpt['scheduler'])

    start = int(global_step / dataset_train.data_size)
    logger.info(f"start: {start}, global_step:{global_step}, gpu_num:{args.gpu_num}")

    for epoch in trange(start, N_iters):
        for iter, data in enumerate(tqdm(train_loader)):
            rays, bc_rgb, auds, pose, gt_img, landmark, index = data

            rays = einops.rearrange(rays, 'b (h w) c -> (b h) w c', h=args.gpu_num)
            bc_rgb = einops.rearrange(bc_rgb, 'b (h w) c -> (b h) w c', h=args.gpu_num)

            auds = einops.repeat(auds, 'b h w c -> (repeat b) h w c', repeat=args.gpu_num)
            pose = einops.repeat(pose, 'b h w -> (repeat b) h w', repeat=args.gpu_num)
            index = einops.repeat(index, 'h -> (repeat h)', repeat=args.gpu_num)

            inp = [rays, bc_rgb, auds, pose, index]

            rgb, rgb0 = network([inp, global_step, dataset_train.data_size])

            """ 对tensor进行一些变换用于计算损失 """
            gt_img = torch.squeeze(gt_img)
            # (h w c)
            landmark = torch.squeeze(landmark)
            # (h w c)
            rgb_face = einops.repeat(rgb0, '(h w) c -> h w c', h=dataset_train.face_size)
            # rgb = einops.rearrange(rgb, '(h w) c -> h w c', h=gt_img.shape[1])
            # rgb0 = einops.rearrange(rgb0, '(h w) c -> h w c', h=gt_img.shape[1])
            # gt_img_vgg = einops.rearrange(gt_img, 'b h w c -> b c h w')
            # rgb_vgg = einops.repeat(rgb, 'h w c -> b c h w', b=1)
            # rgb0_vgg = einops.repeat(rgb0, 'h w c -> b c h w', b=1)

            """ 尝试过的一些损失函数 """
            img_loss = img2mse(rgb, gt_img)
            img_loss0 = img2mse(rgb0, gt_img)
            # loss_rgb, _ = vgg_loss(rgb_vgg, gt_img_vgg)
            # loss_rgb0, _ = vgg_loss(rgb0_vgg, gt_img_vgg)
            # loss_rgb_face, _ = vgg_face_loss(rgb_vgg, gt_img_vgg)
            # loss_rgb0_face, _ = vgg_face_loss(rgb0_vgg, gt_img_vgg)
            loss_lmd = landmark_loss(rgb_face, landmark)

            loss = img_loss  # 为什么去掉了img_loss之后会OOM?
            if loss_lmd is not None:
                loss += loss_lmd

            psnr = mse2psnr(img_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            decay_rate = 0.1
            decay_steps = args.lrate_decay * 1500
            new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))

            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate

            if global_step % args.i_print == 0 and network.training:
                rgb_output = einops.rearrange(rgb, '(h w) c -> h w c', h=256)
                gt_output = einops.rearrange(gt_img, '(h w) c -> h w c', h=256)

                # imageio.imwrite(f'output/test_input/{iter}_rgb.jpg', to8b(rgb_output.detach().cpu().numpy()))
                # imageio.imwrite(f'output/test_input/{iter}_gt.jpg', to8b(gt_output.detach().cpu().numpy()))

                logger.info(
                    f"[TRAIN] epoch: {epoch} Iter: {iter} Loss: {loss.item()} PSNR: {psnr.item()} LR: {new_lrate}")
                torch_writer.add_scalar('loss', loss.item(), global_step=global_step)
                torch_writer.add_scalar('psnr', psnr.item(), global_step=global_step)
                torch_writer.add_scalar('learning_rate', new_lrate, global_step=global_step)

                gt_save = einops.rearrange(gt_output, 'h w c-> c h w').cpu().numpy()
                torch_writer.add_image('input_gt_image', to8b(gt_save), global_step=global_step)

                rgb_save = einops.rearrange(rgb_output, 'h w c -> c h w')
                rgb_cpu = to8b(rgb_save.detach().cpu().numpy())
                torch_writer.add_image("render_images", rgb_cpu, global_step=global_step)

            # 保存模型
            if global_step % args.i_weights == 0 and network.training:
                path = os.path.join(basedir, expname, '{:06d}_head.tar'.format(epoch))
                torch.save({
                    'global_step': global_step,

                    'model_state_dict': network.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }, path)
                logger.info(f'Saved checkpoints at {path} and start to test with network')


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()
