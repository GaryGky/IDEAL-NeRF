import logging
import os
import sys
import time

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = f'{cur_path}/../../'
sys.path.append(root_path)

import cv2
import imageio
from natsort import natsorted
from tqdm import tqdm

from load_audface import load_test_data
from models.audio_net import AudioNet, AudioAttNet
from models.face_nerf import FaceNeRF
from run_nerf_helpers import *
import torch.nn.functional as F

device = torch.device('cuda')
device_torso = torch.device('cuda')
np.random.seed(0)
logger = logging.getLogger('adnerf')
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)
logger.info(f"device: {device}")
DEBUG = True


def render_path(render_poses, aud_paras, bc_img, hwfcxy,
                chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):
    H, W, focal, cx, cy = hwfcxy

    if render_factor != 0:
        # Render downsampled for speed
        H = H // render_factor
        W = W // render_factor
        focal = focal / render_factor

    rgbs = []
    disps = []
    last_weights = []
    rgb_fgs = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        logger.info(f'{i, time.time() - t}')
        t = time.time()
        rgb, disp, acc, last_weight, rgb_fg, _ = render_dynamic_face(
            H, W, focal, cx, cy, chunk=chunk, c2w=c2w[:3,
                                                  :4], aud_para=aud_paras[i], bc_rgb=bc_img,
            **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        last_weights.append(last_weight.cpu().numpy())
        rgb_fgs.append(rgb_fg.cpu().numpy())
        # if i == 0:
        #     print(rgb.shape, disp.shape)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)
    last_weights = np.stack(last_weights, 0)
    rgb_fgs = np.stack(rgb_fgs, 0)

    return rgbs, disps, last_weights, rgb_fgs


def create_nerf(args, ext, dim_aud, device_spec=torch.device('cuda'), with_audatt=False):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(
        args.multires, args.i_embed, device=device_spec)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(
            args.multires_views, args.i_embed, device=device_spec)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = FaceNeRF(D=args.netdepth, W=args.netwidth,
                     input_ch=input_ch, dim_aud=dim_aud,
                     output_ch=output_ch, skips=skips,
                     input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device_spec)
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = FaceNeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                              input_ch=input_ch, dim_aud=dim_aud,
                              output_ch=output_ch, skips=skips,
                              input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device_spec)
        grad_vars += list(model_fine.parameters())

    def network_query_fn(inputs, viewdirs, aud_para, network_fn): \
            return run_network(inputs, viewdirs, aud_para, network_fn,
                               embed_fn=embed_fn, embeddirs_fn=embeddirs_fn, netchunk=args.netchunk)

    # Create optimizer
    optimizer = torch.optim.Adam(
        params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [os.path.join(args.ft_path, f) for f in natsorted(os.listdir(args.ft_path)) if ext in f]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in natsorted(os.listdir(os.path.join(basedir, expname))) if
                 ext in f]

    logger.info(f'Found ckpts:{ckpts}')
    learned_codes_dict = None
    AudNet_state = None
    optimizer_aud_state = None
    AudAttNet_state = None
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        logger.info(f'Reloading from:{ckpt_path}')
        ckpt = torch.load(ckpt_path, map_location=device)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        AudNet_state = ckpt['network_audnet_state_dict']
        optimizer_aud_state = ckpt['optimizer_aud_state_dict']
        if with_audatt:
            AudAttNet_state = ckpt['network_audattnet_state_dict']

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'network_fine': model_fine,
        'N_samples': args.N_samples,
        'network_fn': model,
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
    }

    render_kwargs_test = {
        k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, learned_codes_dict, \
           AudNet_state, optimizer_aud_state, AudAttNet_state


def run_network(inputs, viewdirs, aud_para, fn, embed_fn, embeddirs_fn, netchunk=1024 * 64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)
    # aud = aud_para.unsqueeze(0).expand(inputs_flat.shape[0], -1)
    # embedded = torch.cat((embedded, aud), -1)
    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded, aud_para)
    outputs = torch.reshape(outputs_flat, list(
        inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def render_dynamic_face(H, W, focal, cx, cy, chunk=1024 * 32, rays=None, bc_rgb=None, aud_para=None,
                        c2w=None, ndc=False, near=0., far=1.,
                        use_viewdirs=False, c2w_staticcam=None,
                        **kwargs):
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, focal, c2w, cx, cy, c2w.device)
        bc_rgb = bc_rgb.reshape(-1, 3)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam, cx, cy)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

    sh = rays_d.shape  # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, focal, 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    near, far = near * \
                torch.ones_like(rays_d[..., :1]), far * \
                torch.ones_like(rays_d[..., :1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, bc_rgb, aud_para, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map', 'last_weight', 'rgb_map_fg']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def batchify_rays(rays_flat, bc_rgb, aud_para, chunk=1024 * 32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i + chunk], bc_rgb[i:i + chunk],
                          aud_para, **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn

    def ret(inputs, aud):
        return torch.cat([fn(inputs[i:i + chunk], aud) for i in range(0, inputs.shape[0], chunk)], 0)

    return ret


def render_rays(ray_batch, bc_rgb, aud_para,
                network_fn, network_query_fn,
                N_samples, retraw=False,
                lindisp=False, perturb=0.,
                N_importance=0, network_fine=None,
                white_bkgd=False, raw_noise_std=0.,
                verbose=False, pytest=False):
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples, device=rays_o.device)
    if not lindisp:
        z_vals = near * (1. - t_vals) + far * (t_vals)
    else:
        z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape, device=rays_o.device)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand).to(rays_o.device)
        t_rand[..., -1] = 1.0
        z_vals = lower + (upper - lower) * t_rand
    pts = rays_o[..., None, :] + rays_d[..., None, :] * \
          z_vals[..., :, None]  # [N_rays, N_samples, 3]

    #     raw = run_network(pts)
    raw = network_query_fn(pts, viewdirs, aud_para, network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map, rgb_map_fg = raw2outputs(
        raw, z_vals, rays_d, bc_rgb, raw_noise_std, white_bkgd, pytest=pytest)

    if N_importance > 0:
        rgb_map_0, disp_map_0, acc_map_0, last_weight_0, rgb_map_fg_0 = \
            rgb_map, disp_map, acc_map, weights[..., -1], rgb_map_fg

        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(
            z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
              z_vals[..., :, None]

        run_fn = network_fn if network_fine is None else network_fine
        raw = network_query_fn(pts, viewdirs, aud_para, run_fn)

        rgb_map, disp_map, acc_map, weights, depth_map, rgb_map_fg = raw2outputs(
            raw, z_vals, rays_d, bc_rgb, raw_noise_std, white_bkgd, pytest=pytest)

    ret = {'rgb_map': rgb_map, 'disp_map': disp_map,
           'acc_map': acc_map, 'rgb_map_fg': rgb_map_fg}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]
        ret['last_weight'] = weights[..., -1]
        ret['last_weight0'] = last_weight_0
        ret['rgb_map_fg0'] = rgb_map_fg_0

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            logger.info(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def raw2outputs(raw, z_vals, rays_d, bc_rgb, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """

    def raw2alpha(raw, dists, act_fn=F.relu):
        return 1. - \
               torch.exp(-(act_fn(raw) + 1e-6) * dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([1e10], ).expand(dists[..., :1].shape).to(device)], -1)  # [N_rays, N_samples]

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
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
    weights = alpha * \
              torch.cumprod(
                  torch.cat([torch.ones((alpha.shape[0], 1), device=alpha.device), 1. - alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    rgb_map_fg = torch.sum(weights[:, :-1, None] * rgb[:, :-1, :], -2)

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map, rgb_map_fg


def test_torso():
    parser = config_parser()
    args = parser.parse_args()
    near = args.near
    far = args.far

    print(f'ft_path: {args.ft_path}, aud_file: {args.aud_file}')

    embed_fn, input_ch = get_embedder(3, 0)
    dim_torso_signal = args.dim_aud_body + 2 * input_ch

    poses, auds, bc_img, hwfcxy, aud_ids, torso_pose = \
        load_test_data(args.datadir, args.aud_file, args.test_pose_file, args.testskip, args.test_size, args.aud_start)
    torso_pose = torch.as_tensor(torso_pose).to(device_torso).float()

    H, W, focal, cx, cy = hwfcxy
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    hwfcxy = [H, W, focal, cx, cy]

    # Create torso nerf model
    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, learned_codes, AudNet_state, optimizer_aud_state, AudAttNet_state = create_nerf(
        args, 'head.tar', args.dim_aud, device, True)

    render_kwargs_train_torso, render_kwargs_test_torso, start, grad_vars_torso, optimizer_torso, \
    learned_codes_torso, AudNet_state_torso, optimizer_aud_state_torso, _ = create_nerf(
        args, 'body.tar', dim_torso_signal, device_torso)

    AudNet = AudioNet(args.dim_aud, args.win_size).to(device)
    AudAttNet = AudioAttNet().to(device)
    optimizer_Aud = torch.optim.Adam(params=list(AudNet.parameters()), lr=args.lrate, betas=(0.9, 0.999))

    if AudNet_state is not None:
        AudNet.load_state_dict(AudNet_state)
    if AudAttNet_state is not None:
        logger.info('load audattnet')
        AudAttNet.load_state_dict(AudAttNet_state)
    if optimizer_aud_state is not None:
        optimizer_Aud.load_state_dict(optimizer_aud_state)
    bds_dict = {
        'near': near,
        'far': far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move training data to GPU
    bc_img = torch.Tensor(bc_img).to(device).float() / 255.0
    poses = torch.Tensor(poses).to(device).float()
    auds = torch.Tensor(auds).to(device).float()

    embed_fn, input_ch = get_embedder(3, 0)

    AudNet_torso = AudioNet(args.dim_aud_body, args.win_size).to(device_torso)
    optimizer_Aud_torso = torch.optim.Adam(
        params=list(AudNet_torso.parameters()), lr=args.lrate, betas=(0.9, 0.999))

    if AudNet_state_torso is not None:
        AudNet_torso.load_state_dict(AudNet_state_torso)
    if optimizer_aud_state_torso is not None:
        optimizer_Aud_torso.load_state_dict(optimizer_aud_state_torso)
    bds_dict = {
        'near': near,
        'far': far,
    }
    render_kwargs_train_torso.update(bds_dict)
    render_kwargs_test_torso.update(bds_dict)

    if args.with_test:
        logger.info('RENDER ONLY')
        with torch.no_grad():
            logger.info(f'test poses shape:{poses.shape}')
            smo_half_win = int(args.smo_size / 2)
            auds_val = []
            for i in range(poses.shape[0]):
                left_i = i - smo_half_win
                right_i = i + smo_half_win
                pad_left, pad_right = 0, 0
                if left_i < 0:
                    pad_left = -left_i
                    left_i = 0
                if right_i > poses.shape[0]:
                    pad_right = right_i - poses.shape[0]
                    right_i = poses.shape[0]
                auds_win = auds[left_i:right_i]
                if pad_left > 0:
                    auds_win = torch.cat((torch.zeros_like(auds_win)[:pad_left], auds_win), dim=0)
                if pad_right > 0:
                    auds_win = torch.cat((auds_win, torch.zeros_like(auds_win)[:pad_right]), dim=0)
                auds_win = AudNet(auds_win)
                aud_smo = AudAttNet(auds_win)
                auds_val.append(aud_smo)
            auds_val = torch.stack(auds_val, 0)

            adjust_poses = poses.clone()
            adjust_poses_torso = poses.clone()

            et = pose_to_euler_trans(adjust_poses_torso)
            embed_et = torch.cat(
                (embed_fn(et[:, :3]), embed_fn(et[:, 3:])), dim=-1).to(device_torso)
            signal = torch.cat((auds_val[..., :args.dim_aud_body].to(
                device_torso), embed_et.squeeze()), dim=-1)
            t_start = time.time()

            save_path = args.save_path
            os.makedirs(save_path, exist_ok=True)

            vid_out = cv2.VideoWriter(os.path.join(args.save_path, 'result.avi'),
                                      cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 25, (W, H))

            for j in range(poses.shape[0]):
                rgbs, disps, last_weights, rgb_fgs = \
                    render_path(adjust_poses[j:j + 1], auds_val[j:j + 1],
                                bc_img, hwfcxy, args.chunk, render_kwargs_test)
                rgbs_torso, disps_torso, last_weights_torso, rgb_fgs_torso = \
                    render_path(torso_pose.unsqueeze(
                        0), signal[j:j + 1], bc_img.to(device_torso), hwfcxy, args.chunk, render_kwargs_test_torso)
                rgbs_com = rgbs * last_weights_torso[..., None] + rgb_fgs_torso
                rgb8 = to8b(rgbs_com[0])
                vid_out.write(rgb8[:, :, ::-1])
                filename = os.path.join(save_path, str(aud_ids[j]) + '.jpg')

                if j % 10 == 0:
                    rgb8_fg_torso = to8b(rgb_fgs_torso[0])
                    imageio.imwrite(os.path.join(save_path, str(aud_ids[j]) + '_torso.jpg'), rgb8_fg_torso)
                    imageio.imwrite(filename, rgb8)
                    logger.info(f'finished render{j}')
            logger.info(f'finished render in{time.time() - t_start}')
            vid_out.release()
            return


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    test_torso()
