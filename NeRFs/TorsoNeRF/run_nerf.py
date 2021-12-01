import logging
import os
import time

import cv2
import imageio
from natsort import natsorted
from tqdm import tqdm, trange

from NeRFs.TorsoNeRF.test_torso import render_path
from load_audface import load_audface_data, load_test_data
from models.audio_net import AudioNet, AudioAttNet
from models.face_nerf import FaceNeRF
from run_nerf_helpers import *

device = torch.device('cuda')
device_torso = torch.device('cuda')
np.random.seed(0)
logger = logging.getLogger('adnerf')
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)
DEBUG = True


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn

    def ret(inputs):
        return torch.cat([fn(inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

    return ret


def run_network(inputs, viewdirs, aud_para, fn, embed_fn, embeddirs_fn, netchunk=1024 * 64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)
    aud = aud_para.unsqueeze(0).expand(inputs_flat.shape[0], -1)
    embedded = torch.cat((embedded, aud), -1)
    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(
        inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


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


def render_dynamic_face(H, W, focal, cx, cy, chunk=1024 * 32, rays=None, bc_rgb=None, aud_para=None,
                        c2w=None, ndc=True, near=0., far=1.,
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


def render(H, W, focal, cx, cy, chunk=1024 * 32, rays=None, c2w=None, ndc=True,
           near=0., far=1.,
           use_viewdirs=False, c2w_staticcam=None,
           **kwargs):
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, focal, c2w, cx, cy)
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
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def create_nerf(args, ext, dim_aud, device_spec=torch.device('cuda', 0), with_audatt=False):
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

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        logger.info('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {
        k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, learned_codes_dict, \
           AudNet_state, optimizer_aud_state, AudAttNet_state


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

    # raw = run_network(pts)
    raw = network_query_fn(pts, viewdirs, aud_para, network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map, rgb_map_fg = raw2outputs(
        raw, z_vals, rays_d, bc_rgb, raw_noise_std, white_bkgd, pytest=pytest)

    if N_importance > 0:
        rgb_map_0, disp_map_0, acc_map_0, last_weight_0, rgb_map_fg_0 = rgb_map, disp_map, acc_map, weights[..., -1], rgb_map_fg

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


def train():
    parser = config_parser()
    args = parser.parse_args()

    # Load data
    if args.with_test == 1:
        poses, auds, bc_img, hwfcxy, aud_ids, torso_pose = \
            load_test_data(args.datadir, args.aud_file,
                           args.test_pose_file, args.testskip, args.test_size, args.aud_start)
        torso_pose = torch.as_tensor(torso_pose).to(device_torso).float()
        com_images = np.zeros(1)
    else:
        com_images, poses, auds, bc_img, hwfcxy, sample_rects, i_split = load_audface_data(args.datadir, args.testskip)

    if args.with_test == 0:
        i_train, i_val = i_split

    near = args.near
    far = args.far

    # Cast intrinsics to right types
    H, W, focal, cx, cy = hwfcxy
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    hwfcxy = [H, W, focal, cx, cy]

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, learned_codes, AudNet_state, optimizer_aud_state, AudAttNet_state = create_nerf(
        args, 'head.tar', args.dim_aud, device, True)
    global_step = start

    # 创建AudNet
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

    num_frames = com_images.shape[0]

    embed_fn, input_ch = get_embedder(3, 0)
    dim_torso_signal = args.dim_aud_body + 2 * input_ch
    # Create torso nerf model
    render_kwargs_train_torso, render_kwargs_test_torso, start, grad_vars_torso, optimizer_torso, \
    learned_codes_torso, AudNet_state_torso, optimizer_aud_state_torso, _ = create_nerf(
        args, 'body.tar', dim_torso_signal, device_torso)
    global_step = start

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

    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching
        logger.info('get rays')
        rays = np.stack([get_rays_np(H, W, focal, p, cx, cy)
                         for p in poses[:, :3, :4]], 0)  # [N, ro+rd, H, W, 3]
        logger.info('done, concats')
        # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.concatenate([rays, com_images[:, None]], 1)
        # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])
        rays_rgb = np.stack([rays_rgb[i]
                             for i in i_train], 0)  # train images only
        # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])
        rays_rgb = rays_rgb.astype(np.float32)
        logger.info('shuffle rays')
        np.random.shuffle(rays_rgb)

        logger.info('done')
        i_batch = 0

    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)

    N_iters = args.N_iters + 1
    logger.info('Begin')
    logger.info(f'TRAIN views are{i_train}')
    logger.info(f'VAL views are{i_val}')

    start = start + 1
    for i in trange(start, N_iters):
        time0 = time.time()

        # Sample random ray batch
        if use_batching:
            # Random over all images
            batch = rays_rgb[i_batch:i_batch + N_rand]  # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                logger.info("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0

        else:
            # Random from one image
            img_i = np.random.choice(i_train)
            target_com = torch.as_tensor(imageio.imread(
                com_images[img_i])).to(device).float() / 255.0
            pose = poses[img_i, :3, :4]
            pose_torso = poses[0, :3, :4].to(device_torso)
            rect = sample_rects[img_i]
            aud = auds[img_i]

            smo_half_win = int(args.smo_size / 2)
            left_i = img_i - smo_half_win
            right_i = img_i + smo_half_win
            pad_left, pad_right = 0, 0
            if left_i < 0:
                pad_left = -left_i
                left_i = 0
            if right_i > i_train.shape[0]:
                pad_right = right_i - i_train.shape[0]
                right_i = i_train.shape[0]
            auds_win = auds[left_i:right_i]
            if pad_left > 0:
                auds_win = torch.cat(
                    (torch.zeros_like(auds_win)[:pad_left], auds_win), dim=0)
            if pad_right > 0:
                auds_win = torch.cat(
                    (auds_win, torch.zeros_like(auds_win)[:pad_right]), dim=0)
            auds_win = AudNet(auds_win)
            aud_smo = AudAttNet(auds_win)
            aud_smo_torso = aud_smo.to(device_torso)[..., :args.dim_aud_body]

            et = pose_to_euler_trans(poses[img_i].unsqueeze(0))
            embed_et = torch.cat((embed_fn(et[:, :3]), embed_fn(et[:, 3:])), dim=1).to(device_torso)
            signal = torch.cat((aud_smo_torso, embed_et.squeeze()), dim=-1)
            if N_rand is not None:
                rays_o, rays_d = get_rays(
                    H, W, focal, pose, cx, cy, device)  # (H, W, 3), (H, W, 3)
                rays_o_torso, rays_d_torso = get_rays(
                    H, W, focal, pose_torso, cx, cy, device_torso)

                if i < args.precrop_iters:
                    dH = int(H // 2 * args.precrop_frac)
                    dW = int(W // 2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H // 2 - dH, H // 2 + dH - 1, 2 * dH),
                            torch.linspace(W // 2 - dW, W // 2 + dW - 1, 2 * dW)
                        ), -1)
                    if i == start:
                        logger.info(
                            f"[Config] Center cropping of size {2 * dH} x {2 * dW} is enabled until iter {args.precrop_iters}")
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(
                        0, H - 1, H), torch.linspace(0, W - 1, W)), -1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)
                if args.sample_rate > 0:
                    rect = [0, H / 2, W, H / 2]
                    rect_inds = (coords[:, 0] >= rect[0]) & (
                            coords[:, 0] <= rect[0] + rect[2]) & (
                                        coords[:, 1] >= rect[1]) & (
                                        coords[:, 1] <= rect[1] + rect[3])
                    coords_rect = coords[rect_inds]
                    coords_norect = coords[~rect_inds]
                    rect_num = int(N_rand * float(rect[2]) * rect[3] / H / W)
                    norect_num = N_rand - rect_num
                    select_inds_rect = np.random.choice(
                        coords_rect.shape[0], size=[rect_num], replace=False)  # (N_rand,)
                    # (N_rand, 2)
                    select_coords_rect = coords_rect[select_inds_rect].long()
                    select_inds_norect = np.random.choice(
                        coords_norect.shape[0], size=[norect_num], replace=False)  # (N_rand,)
                    # (N_rand, 2)
                    select_coords_norect = coords_norect[select_inds_norect].long(
                    )
                    select_coords = torch.cat(
                        (select_coords_norect, select_coords_rect), dim=0)

                else:
                    select_inds = np.random.choice(
                        coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                    select_coords = coords[select_inds].long()
                    norect_num = 0

                rays_o = rays_o[select_coords[:, 0],
                                select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0],
                                select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                bc_rgb = bc_img[select_coords[:, 0],
                                select_coords[:, 1]]

                rays_o_torso = rays_o_torso[select_coords[:, 0],
                                            select_coords[:, 1]]  # (N_rand, 3)
                rays_d_torso = rays_d_torso[select_coords[:, 0],
                                            select_coords[:, 1]]  # (N_rand, 3)
                batch_rays_torso = torch.stack([rays_o_torso, rays_d_torso], 0)
                bc_rgb = bc_img[select_coords[:, 0],
                                select_coords[:, 1]]
                bc_rgb_torso = bc_rgb.to(device_torso)

                target_s_com = target_com[select_coords[:, 0],
                                          select_coords[:, 1]]  # (N_rand, 3)

        #####  Core optimization loop  #####
        # 计算head部分
        rgb, disp, acc, last_weight, rgb_fg, extras = \
            render_dynamic_face(H, W, focal, cx, cy, chunk=args.chunk, rays=batch_rays,
                                aud_para=aud_smo, bc_rgb=bc_rgb,
                                verbose=i < 10, retraw=True,
                                **render_kwargs_train)
        # 计算torso部分
        rgb_torso, disp_torso, acc_torso, last_weight_torso, rgb_fg_torso, extras_torso = \
            render_dynamic_face(H, W, focal, cx, cy, chunk=args.chunk, rays=batch_rays_torso,
                                aud_para=signal, bc_rgb=bc_rgb_torso,
                                verbose=i < 10, retraw=True,
                                **render_kwargs_train_torso)
        rgb_com = rgb * last_weight_torso.to(device)[..., None] + rgb_fg_torso.to(device)

        optimizer_torso.zero_grad()
        img_loss_com = img2mse(rgb_com, target_s_com)
        trans = extras['raw'][..., -1]
        split_weight = float(1.0)
        loss = img_loss_com
        psnr = mse2psnr(img_loss_com)

        if 'rgb0' in extras_torso:
            rgb_com0 = extras['rgb0'] * \
                       extras_torso['last_weight0'].to(
                           device)[..., None] + extras_torso['rgb_map_fg0'].to(device)
            img_loss0 = img2mse(rgb_com0, target_s_com)
            loss = loss + img_loss0

        loss.backward()
        optimizer_torso.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1500
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        # print('cur_rate', new_lrate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        for param_group in optimizer_Aud.param_groups:
            param_group['lr'] = new_lrate

        for param_group in optimizer_torso.param_groups:
            param_group['lr'] = new_lrate

        for param_group in optimizer_Aud_torso.param_groups:
            param_group['lr'] = new_lrate
        ################################

        # Rest is logging
        if i % args.i_weights == 0:
            path = os.path.join(basedir, expname, '{:06d}_head.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'network_audnet_state_dict': AudNet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'optimizer_aud_state_dict': optimizer_Aud.state_dict(),
                'network_audattnet_state_dict': AudAttNet.state_dict(),
            }, path)

            path = os.path.join(basedir, expname, '{:06d}_body.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train_torso['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train_torso['network_fine'].state_dict(),
                'network_audnet_state_dict': AudNet_torso.state_dict(),
                'optimizer_state_dict': optimizer_torso.state_dict(),
                'optimizer_aud_state_dict': optimizer_Aud_torso.state_dict(),
            }, path)
            logger.info(f'Saved checkpoints at{path}')

        if i % args.i_testset == 0 and i > 0:
            testsavedir = os.path.join(
                basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            logger.info(f'test poses shape{poses[i_val].shape}')

            aud_torso = AudNet(
                auds[i_val])[..., :args.dim_aud_body].to(device_torso)
            et = pose_to_euler_trans(poses[i_val])
            embed_et = torch.cat(
                (embed_fn(et[:, :3]), embed_fn(et[:, 3:])), dim=1).to(device_torso)
            signal = torch.cat((aud_torso, embed_et.squeeze()), dim=-1)

            auds_val = AudNet(auds[i_val])
            with torch.no_grad():
                for j in range(auds_val.shape[0]):
                    rgbs, disps, last_weights, rgb_fgs = \
                        render_path(poses[i_val][j:j + 1], auds_val[j:j + 1],
                                    bc_img, hwfcxy, args.chunk, render_kwargs_test)
                    rgbs_torso, disps_torso, last_weights_torso, rgb_fgs_torso = \
                        render_path(poses[0].to(device_torso).unsqueeze(0),
                                    signal[j:j + 1], bc_img.to(
                                device_torso), hwfcxy, args.chunk, render_kwargs_test_torso)
                    rgbs_com = rgbs * \
                               last_weights_torso[..., None] + rgb_fgs_torso
                    rgb8 = to8b(rgbs_com[0])
                    filename = os.path.join(
                        testsavedir, '{:03d}.jpg'.format(j))
                    imageio.imwrite(filename, rgb8)
            logger.info('Saved test set')

        if i % args.i_print == 0:
            tqdm.write(
                f"[TRAIN] Iter: {i} Loss: {img_loss_com.item()}  PSNR: {psnr.item()}")

        global_step += 1


#
# if __name__ == '__main__':
#     torch.set_default_tensor_type('torch.cuda.FloatTensor')
#     pass

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
    dists = torch.cat([dists, torch.Tensor([1e10], device=z_vals.device).expand(
        dists[..., :1].shape)], -1)  # [N_rays, N_samples]

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
