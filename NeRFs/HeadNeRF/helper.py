import face_alignment
import numpy as np
import torch
import torch.nn as nn
import einops
import torch.nn.functional as F
import os, sys

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = f'{cur_path}/../..'
sys.path.append(root_path)

from utils.landmark_utils import get_lms_features


def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    """实验目录部分的参数"""
    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument("--expname", type=str, help='experiment name')
    parser.add_argument("--basedir", type=str, help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./dataset/Obama', help='input data directory')
    parser.add_argument("--vis_path", type=str, default='./dataset/Obama/run', help='input data directory')
    parser.add_argument("--save_path", type=str, default='output/render/Obama-Noah/', help='output render directory')
    parser.add_argument("--evalExpr_path", type=str, help='模型测试时使用的expression路径')

    # 模型训练的参数
    parser.add_argument("--mouth_rays", type=int, default=0, help='大于0 表示对嘴部进行highlight')
    parser.add_argument("--torso_rays", type=int, default=0, help='大于0 表示对Torso 进行highlight')
    parser.add_argument("--dim_expr", type=int, default=0, help="exp特征的维度")
    parser.add_argument("--dim_aud", type=int, default=0, help='dimension of audio features for NeRF')


    parser.add_argument("--lc_weight", type=float, default=0.0005, help='latent code的权重')
    parser.add_argument("--gt_dirs", type=str, default='head_imgs', help='用于采集射线图像的来源')
    parser.add_argument("--gpu_num", type=int, default=0, help='select gpu numbers to use')
    parser.add_argument("--num_work", type=int, default=3, help='how many workers use to load data')
    parser.add_argument("--batch_size", type=int, default=4, help='batch_size: load batch_size images one time')
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8,
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=2048,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=8e-4,
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=500,
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024 * 8,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024 * 64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--use_batching", action='store_false', help='是否使用视角向量')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--N_iters", type=int, default=90, help='number of iterations')  # epoch

    """ 神经渲染参数 """
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=128,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_false',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')

    # dataset options
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    # blender flags
    parser.add_argument("--white_bkgd", action='store_false',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')

    # face flags
    parser.add_argument("--with_test", type=int, default=0,
                        help='whether to use test set')
    parser.add_argument("--sample_rate", type=float, default=0.95,
                        help="sample rate in a bounding box")
    parser.add_argument("--near", type=float, default=0.3,
                        help="near sampling plane")
    parser.add_argument("--far", type=float, default=0.9,
                        help="far sampling plane")
    parser.add_argument("--test_file", type=str, help='test file')
    parser.add_argument("--aud_file", type=str, default='aud.npy',
                        help='test audio deepspeech file')
    parser.add_argument("--win_size", type=int, default=16,
                        help="windows size of audio feature")
    parser.add_argument("--smo_size", type=int, default=8,
                        help="window size for smoothing audio features")
    parser.add_argument('--nosmo_iters', type=int, default=300000,
                        help='number of iterations befor applying smoothing on audio features')

    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')

    """保存模型和日志相关的参数"""
    parser.add_argument("--i_print", type=int, default=10,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img", type=int, default=500,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=5000, help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=1000, help='frequency of testset saving')
    parser.add_argument("--i_video", type=int, default=5000, help='frequency of render_poses video saving')

    return parser


parser = config_parser()
args = parser.parse_args()


# torch.autograd.set_detect_anomaly(True)


def img2mse(x, y): return F.mse_loss(x, y)


def mse2psnr(x): return -10. * torch.log(x) / torch.log(torch.Tensor([10.]).cuda())


def to8b(x): return (255 * np.clip(x, 0, 1)).astype(np.uint8)


def to8b_tensor(x): return 255 * torch.clip(x, 0, 1)


def lmd_loss(pred, target):
    target = target.detach()
    pred = pred.cpu()

    face_detector = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cpu')
    pred_lms = face_detector.get_landmarks(pred)

    if pred_lms is None or len(pred_lms) == 0:
        return 0

    return F.mse_loss(get_lms_features(pred_lms[0]), target)


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs).cuda()
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs).cuda()

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x.cuda() * freq.cuda()))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1).cuda()


def get_embedder(multires, i=0, input_dims=3):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)

    def embed(x, eo=embedder_obj): return eo.embed(x)

    return embed, embedder_obj.out_dim


# Ray helpers
def get_rays(H, W, focal, c2w, cx=None, cy=None):
    # pytorch's meshgrid has indexing='ij'
    i, j = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H))
    i = i.t()
    j = j.t()
    if cx is None:
        cx = W * .5
    if cy is None:
        cy = H * .5
    dirs = torch.stack([(i - cx) / focal, -(j - cy) / focal, -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1. / (W / (2. * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1. / (H / (2. * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1. / (W / (2. * focal)) * \
         (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    d1 = -1. / (H / (2. * focal)) * \
         (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    # (batch, len(bins))
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples).cuda()
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples]).cuda()

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


# Inverts a pose or extrinsic matrix
def invert(mat):
    rot = mat[..., :3, :3]
    trans = mat[..., :3, 3, None]
    rot_t = torch.transpose(rot, 2, 1)
    trans_t = -1 * torch.transpose(rot, 2, 1) @ trans
    return torch.cat([rot_t, trans_t], -1)


def make_indices(pts, attention_poses, intrinsic, H, W):
    eps = einops.repeat(torch.Tensor([1.0]).to(pts.device), "w -> h w", h=pts.shape[0], w=1)
    hom_points = torch.cat([pts, eps], -1)
    extrinsic = invert(attention_poses[None, ...])[:, :3]
    focal = intrinsic[0, 0]

    hom_points = einops.repeat(hom_points[None, ...], "h w c -> h w c", h=extrinsic.shape[0], w=hom_points.shape[0],
                               c=hom_points.shape[1])

    pt_camera = torch.matmul(hom_points, torch.transpose(extrinsic, 2, 1))

    pt_camera = focal / pt_camera[:, :, 2][..., None] * pt_camera

    intrinsic = torch.transpose(torch.tensor(intrinsic, dtype=torch.float32).to(pt_camera.device), 0, 1)
    final = 1.0 / focal * (torch.matmul(pt_camera, intrinsic))
    final = torch.flip(final, dims=[2])[..., 1:]
    final = (torch.tensor([0., W]).to(final.device) - final) * torch.tensor([-1., 1.]).to(final.device)
    final = torch.round(final)
    min_index_zero = torch.zeros(final.shape).long()
    max_index_H = min_index_zero + (H - 1)
    final = torch.minimum(final, max_index_H.long().to(final.device))
    final = torch.maximum(final, min_index_zero.to(final.device))

    return final.long()


def gather_indices(pts, attention_poses, intrinsic, images_features):
    H, W = images_features.shape[2:4]
    H = int(H)
    W = int(W)
    indices = make_indices(pts, attention_poses, intrinsic, H, W)
    indices = torch.squeeze(indices)

    features = torch.squeeze(images_features)
    features = einops.rearrange(features, "c w h ->w h c")

    indices = torch.clamp(indices, min=0, max=H - 1)

    if torch.max(indices) >= 450 or torch.min(indices) < 0:
        raise Exception(f"gather_indices index error:{torch.max(indices)} || {torch.min(indices)} || {indices}")

    features = features[indices[:, 0], indices[:, 1]]

    return torch.cat([features, indices.type(torch.float32)], -1)  # (1024*64, 196)


def write_config(args):
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


if __name__ == '__main__':
    attention_embed_func, attention_embed_out_dim = get_embedder(3, 0)
    print(attention_embed_func)
    print(attention_embed_out_dim)
