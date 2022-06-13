# Copyright 2021 Alex Yu
# First, install svox2
# Then, python opt.py <path_to>/nerf_synthetic/<scene> -t ckpt/<some_name>
# or use launching script:       sh launch.sh <EXP_NAME> <GPU> <DATA_DIR>

import torch
import torch.optim
import torch.nn.functional as F
import svox2
import json
import imageio
import os
from os import path
import shutil
import gc
import numpy as np
import math
import argparse
import cv2
from util.dataset import datasets
from util.util import get_expon_lr_func
from util import config_util

from warnings import warn
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import cv2

from icecream import ic

# from style_transfer_losses import StyleTransferLosses
from nnfm_loss import NNFMLoss, match_colors_for_image_set


device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
config_util.define_common_args(parser)


#### ARF parameters
parser.add_argument("--init_ckpt", type=str, default="", help="initial checkpoint to load")
parser.add_argument("--style", type=str, help="path to style image")
parser.add_argument("--content_weight", type=float, default=5e-3, help="content loss weight")
parser.add_argument("--img_tv_weight", type=float, default=1, help="image tv loss weight")
parser.add_argument(
    "--vgg_block",
    type=int,
    default=2,
    help="vgg block for extracting feature maps",
)
parser.add_argument(
    "--reset_basis_dim",
    type=int,
    default=1,
    help="whether to reset the number of spherical harmonics basis to this specified number",
)
parser.add_argument(
    "--mse_num_epoches",
    type=int,
    default=2,
    help="epoches for mse loss optimization",
)
parser.add_argument(
    "--nnfm_num_epoches",
    type=int,
    default=10,
    help="epoches for running style transfer",
)
parser.add_argument("--no_pre_ct", action="store_true", default=False)
parser.add_argument("--no_post_ct", action="store_true", default=False)
#### END of ARF parameters


group = parser.add_argument_group("general")
group.add_argument(
    "--train_dir",
    "-t",
    type=str,
    default="ckpt",
    help="checkpoint and logging directory",
)

group.add_argument(
    "--reso",
    type=str,
    default="[[256, 256, 256], [512, 512, 512]]",
    help="List of grid resolution (will be evaled as json);"
    "resamples to the next one every upsamp_every iters, then "
    + "stays at the last one; "
    + "should be a list where each item is a list of 3 ints or an int",
)

group.add_argument(
    "--upsamp_every",
    type=int,
    default=3 * 12800,
    help="upsample the grid every x iters",
)
group.add_argument("--init_iters", type=int, default=0, help="do not upsample for first x iters")
group.add_argument(
    "--upsample_density_add",
    type=float,
    default=0.0,
    help="add the remaining density by this amount when upsampling",
)

group.add_argument(
    "--basis_type",
    choices=["sh", "3d_texture", "mlp"],
    default="sh",
    help="Basis function type",
)

group.add_argument(
    "--basis_reso",
    type=int,
    default=32,
    help="basis grid resolution (only for learned texture)",
)
group.add_argument("--sh_dim", type=int, default=9, help="SH/learned basis dimensions (at most 10)")

group.add_argument(
    "--mlp_posenc_size",
    type=int,
    default=4,
    help="Positional encoding size if using MLP basis; 0 to disable",
)
group.add_argument("--mlp_width", type=int, default=32, help="MLP width if using MLP basis")

group.add_argument(
    "--background_nlayers",
    type=int,
    default=0,  # 32,
    help="Number of background layers (0=disable BG model)",
)
group.add_argument("--background_reso", type=int, default=512, help="Background resolution")


group = parser.add_argument_group("optimization")
group.add_argument(
    "--n_iters",
    type=int,
    default=10 * 12800,
    help="total number of iters to optimize for",
)
group.add_argument(
    "--batch_size",
    type=int,
    default=5000,
    # 100000,
    #      2000,
    help="batch size",
)


# TODO: make the lr higher near the end
group.add_argument(
    "--sigma_optim",
    choices=["sgd", "rmsprop"],
    default="rmsprop",
    help="Density optimizer",
)
group.add_argument("--lr_sigma", type=float, default=3e1, help="SGD/rmsprop lr for sigma")
group.add_argument("--lr_sigma_final", type=float, default=5e-2)
group.add_argument("--lr_sigma_decay_steps", type=int, default=250000)
group.add_argument(
    "--lr_sigma_delay_steps",
    type=int,
    default=15000,
    help="Reverse cosine steps (0 means disable)",
)
group.add_argument("--lr_sigma_delay_mult", type=float, default=1e-2)  # 1e-4)#1e-4)


group.add_argument("--sh_optim", choices=["sgd", "rmsprop"], default="rmsprop", help="SH optimizer")
group.add_argument("--lr_sh", type=float, default=1e-2, help="SGD/rmsprop lr for SH")
group.add_argument("--lr_sh_final", type=float, default=5e-6)
group.add_argument("--lr_sh_decay_steps", type=int, default=250000)
group.add_argument(
    "--lr_sh_delay_steps",
    type=int,
    default=0,
    help="Reverse cosine steps (0 means disable)",
)
group.add_argument("--lr_sh_delay_mult", type=float, default=1e-2)

group.add_argument(
    "--lr_fg_begin_step",
    type=int,
    default=0,
    help="Foreground begins training at given step number",
)

# BG LRs
group.add_argument(
    "--bg_optim",
    choices=["sgd", "rmsprop"],
    default="rmsprop",
    help="Background optimizer",
)
group.add_argument("--lr_sigma_bg", type=float, default=3e0, help="SGD/rmsprop lr for background")
group.add_argument(
    "--lr_sigma_bg_final",
    type=float,
    default=3e-3,
    help="SGD/rmsprop lr for background",
)
group.add_argument("--lr_sigma_bg_decay_steps", type=int, default=250000)
group.add_argument(
    "--lr_sigma_bg_delay_steps",
    type=int,
    default=0,
    help="Reverse cosine steps (0 means disable)",
)
group.add_argument("--lr_sigma_bg_delay_mult", type=float, default=1e-2)

group.add_argument("--lr_color_bg", type=float, default=1e-1, help="SGD/rmsprop lr for background")
group.add_argument(
    "--lr_color_bg_final",
    type=float,
    default=5e-6,  # 1e-4,
    help="SGD/rmsprop lr for background",
)
group.add_argument("--lr_color_bg_decay_steps", type=int, default=250000)
group.add_argument(
    "--lr_color_bg_delay_steps",
    type=int,
    default=0,
    help="Reverse cosine steps (0 means disable)",
)
group.add_argument("--lr_color_bg_delay_mult", type=float, default=1e-2)
# END BG LRs

group.add_argument(
    "--basis_optim",
    choices=["sgd", "rmsprop"],
    default="rmsprop",
    help="Learned basis optimizer",
)
group.add_argument("--lr_basis", type=float, default=1e-6, help="SGD/rmsprop lr for SH")  # 2e6,
group.add_argument("--lr_basis_final", type=float, default=1e-6)
group.add_argument("--lr_basis_decay_steps", type=int, default=250000)
group.add_argument(
    "--lr_basis_delay_steps",
    type=int,
    default=0,  # 15000,
    help="Reverse cosine steps (0 means disable)",
)
group.add_argument("--lr_basis_begin_step", type=int, default=0)  # 4 * 12800)
group.add_argument("--lr_basis_delay_mult", type=float, default=1e-2)

group.add_argument("--rms_beta", type=float, default=0.95, help="RMSProp exponential averaging factor")

group.add_argument("--print_every", type=int, default=20, help="print every")
group.add_argument("--save_every", type=int, default=1, help="save every x epochs")
group.add_argument("--eval_every", type=int, default=1, help="evaluate every x epochs")

group.add_argument("--init_sigma", type=float, default=0.1, help="initialization sigma")
group.add_argument("--init_sigma_bg", type=float, default=0.1, help="initialization sigma (for BG)")

# Extra logging
group.add_argument("--log_mse_image", action="store_true", default=False)
group.add_argument("--log_depth_map", action="store_true", default=False)
group.add_argument(
    "--log_depth_map_use_thresh",
    type=float,
    default=None,
    help="If specified, uses the Dex-neRF version of depth with given thresh; else returns expected term",
)


group = parser.add_argument_group("misc experiments")
group.add_argument(
    "--thresh_type",
    choices=["weight", "sigma"],
    default="weight",
    help="Upsample threshold type",
)
group.add_argument(
    "--weight_thresh",
    type=float,
    default=0.0005 * 512,
    #  default=0.025 * 512,
    help="Upsample weight threshold; will be divided by resulting z-resolution",
)
group.add_argument("--density_thresh", type=float, default=5.0, help="Upsample sigma threshold")
group.add_argument(
    "--background_density_thresh",
    type=float,
    default=1.0 + 1e-9,
    help="Background sigma threshold for sparsification",
)
group.add_argument(
    "--max_grid_elements",
    type=int,
    default=44_000_000,
    help="Max items to store after upsampling " "(the number here is given for 22GB memory)",
)

group.add_argument(
    "--tune_mode",
    action="store_true",
    default=False,
    help="hypertuning mode (do not save, for speed)",
)
group.add_argument(
    "--tune_nosave",
    action="store_true",
    default=False,
    help="do not save any checkpoint even at the end",
)


group = parser.add_argument_group("losses")
# Foreground TV
group.add_argument("--lambda_tv", type=float, default=1e-5)
group.add_argument("--tv_sparsity", type=float, default=0.01)
group.add_argument(
    "--tv_logalpha",
    action="store_true",
    default=False,
    help="Use log(1-exp(-delta * sigma)) as in neural volumes",
)

group.add_argument("--lambda_tv_sh", type=float, default=1e-3)
group.add_argument("--tv_sh_sparsity", type=float, default=0.01)

group.add_argument("--lambda_tv_lumisphere", type=float, default=0.0)  # 1e-2)#1e-3)
group.add_argument("--tv_lumisphere_sparsity", type=float, default=0.01)
group.add_argument("--tv_lumisphere_dir_factor", type=float, default=0.0)

group.add_argument("--tv_decay", type=float, default=1.0)

group.add_argument("--lambda_l2_sh", type=float, default=0.0)  # 1e-4)
group.add_argument(
    "--tv_early_only",
    type=int,
    default=1,
    help="Turn off TV regularization after the first split/prune",
)

group.add_argument(
    "--tv_contiguous",
    type=int,
    default=1,
    help="Apply TV only on contiguous link chunks, which is faster",
)
# End Foreground TV

group.add_argument(
    "--lambda_sparsity",
    type=float,
    default=0.0,
    help="Weight for sparsity loss as in SNeRG/PlenOctrees " + "(but applied on the ray)",
)
group.add_argument(
    "--lambda_beta",
    type=float,
    default=0.0,
    help="Weight for beta distribution sparsity loss as in neural volumes",
)


# Background TV
group.add_argument("--lambda_tv_background_sigma", type=float, default=1e-2)
group.add_argument("--lambda_tv_background_color", type=float, default=1e-2)

group.add_argument("--tv_background_sparsity", type=float, default=0.01)
# End Background TV

# Basis TV
group.add_argument(
    "--lambda_tv_basis",
    type=float,
    default=0.0,
    help="Learned basis total variation loss",
)
# End Basis TV

group.add_argument("--weight_decay_sigma", type=float, default=1.0)
group.add_argument("--weight_decay_sh", type=float, default=1.0)

group.add_argument("--lr_decay", action="store_true", default=True)

group.add_argument(
    "--n_train",
    type=int,
    default=None,
    help="Number of training images. Defaults to use all avaiable.",
)

group.add_argument(
    "--nosphereinit",
    action="store_true",
    default=False,
    help="do not start with sphere bounds (please do not use for 360)",
)

args = parser.parse_args()
config_util.maybe_merge_config_file(args)

assert args.lr_sigma_final <= args.lr_sigma, "lr_sigma must be >= lr_sigma_final"
assert args.lr_sh_final <= args.lr_sh, "lr_sh must be >= lr_sh_final"
assert args.lr_basis_final <= args.lr_basis, "lr_basis must be >= lr_basis_final"

os.makedirs(args.train_dir, exist_ok=True)
summary_writer = SummaryWriter(args.train_dir)

reso_list = json.loads(args.reso)
reso_id = 0

with open(path.join(args.train_dir, "args.json"), "w") as f:
    json.dump(args.__dict__, f, indent=2)
    # Changed name to prevent errors
    shutil.copyfile(__file__, path.join(args.train_dir, "opt_frozen.py"))

torch.manual_seed(20200823)
np.random.seed(20200823)

factor = 1
dset = datasets[args.dataset_type](
    args.data_dir,
    split="train",
    device=device,
    factor=factor,
    n_images=args.n_train,
    **config_util.build_data_options(args),
)

assert dset.rays.origins.shape == (dset.n_images * dset.h * dset.w, 3)
assert dset.rays.dirs.shape == (dset.n_images * dset.h * dset.w, 3)

if args.background_nlayers > 0 and not dset.should_use_background:
    warn("Using a background model for dataset type " + str(type(dset)) + " which typically does not use background")

assert os.path.isfile(args.init_ckpt), "must specify a initial checkpoint"
grid = svox2.SparseGrid.load(args.init_ckpt, device=device, reset_basis_dim=args.reset_basis_dim)
ic("Loaded ckpt: ", args.init_ckpt)
ic(grid.basis_dim)

optim_basis_mlp = None

if grid.basis_type == svox2.BASIS_TYPE_3D_TEXTURE:
    grid.reinit_learned_bases(init_type="sh")
    #  grid.reinit_learned_bases(init_type='fourier')
    #  grid.reinit_learned_bases(init_type='sg', upper_hemi=True)
    #  grid.basis_data.data.normal_(mean=0.28209479177387814, std=0.001)

elif grid.basis_type == svox2.BASIS_TYPE_MLP:
    # MLP!
    optim_basis_mlp = torch.optim.Adam(grid.basis_mlp.parameters(), lr=args.lr_basis)

grid.requires_grad_(True)
config_util.setup_render_opts(grid.opt, args)
print("Render options", grid.opt)

gstep_id_base = 0

lr_sigma_func = get_expon_lr_func(
    args.lr_sigma,
    args.lr_sigma_final,
    args.lr_sigma_delay_steps,
    args.lr_sigma_delay_mult,
    args.lr_sigma_decay_steps,
)
lr_sh_func = get_expon_lr_func(
    args.lr_sh,
    args.lr_sh_final,
    args.lr_sh_delay_steps,
    args.lr_sh_delay_mult,
    args.lr_sh_decay_steps,
)
lr_basis_func = get_expon_lr_func(
    args.lr_basis,
    args.lr_basis_final,
    args.lr_basis_delay_steps,
    args.lr_basis_delay_mult,
    args.lr_basis_decay_steps,
)
lr_sigma_bg_func = get_expon_lr_func(
    args.lr_sigma_bg,
    args.lr_sigma_bg_final,
    args.lr_sigma_bg_delay_steps,
    args.lr_sigma_bg_delay_mult,
    args.lr_sigma_bg_decay_steps,
)
lr_color_bg_func = get_expon_lr_func(
    args.lr_color_bg,
    args.lr_color_bg_final,
    args.lr_color_bg_delay_steps,
    args.lr_color_bg_delay_mult,
    args.lr_color_bg_decay_steps,
)
lr_sigma_factor = 1.0
lr_sh_factor = 1.0
lr_basis_factor = 1.0

last_upsamp_step = args.init_iters

if args.enable_random:
    warn("Randomness is enabled for training (normal for LLFF & scenes with background)")


###### resize style image such that its long side matches the long side of content images
style_img = imageio.imread(args.style).astype(np.float32) / 255.0
style_h, style_w = style_img.shape[:2]
content_long_side = max([dset.w, dset.h])
if style_h > style_w:
    style_img = cv2.resize(
        style_img,
        (int(content_long_side / style_h * style_w), content_long_side),
        interpolation=cv2.INTER_AREA,
    )
else:
    style_img = cv2.resize(
        style_img,
        (content_long_side, int(content_long_side / style_w * style_h)),
        interpolation=cv2.INTER_AREA,
    )
style_img = cv2.resize(
    style_img,
    (style_img.shape[1] // 2, style_img.shape[0] // 2),
    interpolation=cv2.INTER_AREA,
)
imageio.imwrite(
    os.path.join(args.train_dir, "style_image.png"),
    np.clip(style_img * 255.0, 0.0, 255.0).astype(np.uint8),
)
style_img = torch.from_numpy(style_img).to(device=device)
ic("Style image: ", args.style, style_img.shape)


global_start_time = datetime.now()

if not args.no_pre_ct:
    dset.rays.gt, color_tf = match_colors_for_image_set(dset.rays.gt, style_img)
    grid.apply_ct(color_tf.detach().cpu().numpy())

epoch_id = 0
epoch_size = None
batches_per_epoch = None
batch_size = None

nnfm_loss_fn = NNFMLoss(device=device)

while True:
    def train_step(optim_type):
        ic("Training epoch: ", epoch_id, epoch_size, batches_per_epoch, batch_size, optim_type)
        pbar = tqdm(enumerate(range(0, epoch_size, batch_size)), total=batches_per_epoch)
        for iter_id, batch_begin in pbar:
            stats = {}

            gstep_id = iter_id + gstep_id_base
            lr_sigma = lr_sigma_func(gstep_id) * lr_sigma_factor
            lr_sh = lr_sh_func(gstep_id) * lr_sh_factor
            lr_basis = lr_basis_func(gstep_id - args.lr_basis_begin_step) * lr_basis_factor
            lr_sigma_bg = lr_sigma_bg_func(gstep_id - args.lr_basis_begin_step) * lr_basis_factor
            lr_color_bg = lr_color_bg_func(gstep_id - args.lr_basis_begin_step) * lr_basis_factor

            if optim_type == "ray":
                """low frequency transfer"""
                batch_end = min(batch_begin + args.batch_size, epoch_size)
                batch_origins = dset.rays.origins[batch_begin:batch_end].to(device)
                batch_dirs = dset.rays.dirs[batch_begin:batch_end].to(device)
                rgb_gt = dset.rays.gt[batch_begin:batch_end].to(device)
                rays = svox2.Rays(batch_origins, batch_dirs)

                rgb_pred = grid.volume_render_fused(
                    rays,
                    rgb_gt,
                    beta_loss=args.lambda_beta,
                    sparsity_loss=args.lambda_sparsity,
                    randomize=args.enable_random,
                    is_rgb_gt=True,
                    reset_grad_indexers=True,
                )
                mse = F.mse_loss(rgb_gt, rgb_pred)
                # Stats
                psnr = -10.0 * math.log10(mse.detach().item())
                stats["psnr"] = psnr
            elif optim_type == "image":
                num_views, view_height, view_width = dset.n_images, dset.h, dset.w
                img_id = np.random.randint(low=0, high=num_views)
                rays = svox2.Rays(
                    dset.rays.origins.view(num_views, view_height * view_width, 3)[img_id].to(device),
                    dset.rays.dirs.view(num_views, view_height * view_width, 3)[img_id].to(device),
                )
                def compute_image_loss():
                    with torch.no_grad():
                        cam = svox2.Camera(
                            dset.c2w[img_id].to(device=device),
                            dset.intrins.get("fx", img_id),
                            dset.intrins.get("fy", img_id),
                            dset.intrins.get("cx", img_id),
                            dset.intrins.get("cy", img_id),
                            width=view_width,
                            height=view_height,
                            ndc_coeffs=dset.ndc_coeffs,
                        )
                        rgb_pred = grid.volume_render_image(cam, use_kernel=True)
                        rgb_gt = dset.rays.gt.view(num_views, view_height, view_width, 3)[img_id].to(
                            device
                        )
                        rgb_gt = rgb_gt.permute(2, 0, 1).unsqueeze(0).contiguous()
                        rgb_pred = rgb_pred.permute(2, 0, 1).unsqueeze(0).contiguous()

                    rgb_pred.requires_grad_(True)
                    w_variance = torch.mean(torch.pow(rgb_pred[:, :, :, :-1] - rgb_pred[:, :, :, 1:], 2))
                    h_variance = torch.mean(torch.pow(rgb_pred[:, :, :-1, :] - rgb_pred[:, :, 1:, :], 2))
                    img_tv_loss = args.img_tv_weight * (h_variance + w_variance) / 2.0
                    loss_dict = nnfm_loss_fn(
                        F.interpolate(
                            rgb_pred,
                            size=None,
                            scale_factor=0.5,
                            mode="bilinear",
                        ),
                        style_img.permute(2, 0, 1).unsqueeze(0),
                        blocks=[
                            args.vgg_block,
                        ],
                        loss_names=["nnfm_loss", "content_loss"],
                        contents=F.interpolate(
                            rgb_gt,
                            size=None,
                            scale_factor=0.5,
                            mode="bilinear",
                        ),
                    )
                    loss_dict["content_loss"] *= args.content_weight
                    loss_dict["img_tv_loss"] = img_tv_loss
                    loss = sum(list(loss_dict.values()))
                    loss.backward()
                    rgb_pred_grad = rgb_pred.grad.squeeze(0).permute(1, 2, 0).contiguous().clone().detach().view(-1, 3)
                    return rgb_pred_grad, loss_dict

                rgb_pred_grad, loss_dict = compute_image_loss()
                rgb_pred = []
                grid.alloc_grad_indexers()
                for view_batch_start in range(0, view_height * view_width, args.batch_size):
                    rgb_pred_patch = grid.volume_render_fused(
                        rays[view_batch_start : view_batch_start + args.batch_size],
                        rgb_pred_grad[view_batch_start : view_batch_start + args.batch_size],
                        beta_loss=args.lambda_beta,
                        sparsity_loss=args.lambda_sparsity,
                        randomize=args.enable_random,
                        is_rgb_gt=False,
                        reset_grad_indexers=False,
                    )
                    rgb_pred.append(rgb_pred_patch.clone().detach())
                rgb_pred = torch.cat(rgb_pred, dim=0).reshape(view_height, view_width, 3)

                # Stats
                for x in loss_dict:
                    stats[x] = loss_dict[x].item()

            if (iter_id + 1) % args.print_every == 0:
                log_str = ""
                for stat_name in stats:
                    summary_writer.add_scalar(stat_name, stats[stat_name], global_step=gstep_id)
                    log_str += "{:.4f} ".format(stats[stat_name])
                pbar.set_description(f"{gstep_id} {log_str}")

                summary_writer.add_scalar("lr_sh", lr_sh, global_step=gstep_id)
                summary_writer.add_scalar("lr_sigma", lr_sigma, global_step=gstep_id)
                if grid.basis_type == svox2.BASIS_TYPE_3D_TEXTURE:
                    summary_writer.add_scalar("lr_basis", lr_basis, global_step=gstep_id)
                if grid.use_background:
                    summary_writer.add_scalar("lr_sigma_bg", lr_sigma_bg, global_step=gstep_id)
                    summary_writer.add_scalar("lr_color_bg", lr_color_bg, global_step=gstep_id)

            if args.weight_decay_sh < 1.0:
                grid.sh_data.data *= args.weight_decay_sigma
            if args.weight_decay_sigma < 1.0:
                grid.density_data.data *= args.weight_decay_sh

            # Apply TV/Sparsity regularizers
            if args.lambda_tv > 0.0:
                #  with Timing("tv_inpl"):
                grid.inplace_tv_grad(
                    grid.density_data.grad,
                    scaling=args.lambda_tv,
                    sparse_frac=args.tv_sparsity,
                    logalpha=args.tv_logalpha,
                    ndc_coeffs=dset.ndc_coeffs,
                    contiguous=args.tv_contiguous,
                )
            if args.lambda_tv_sh > 0.0:
                #  with Timing("tv_color_inpl"):
                grid.inplace_tv_color_grad(
                    grid.sh_data.grad,
                    scaling=args.lambda_tv_sh,
                    sparse_frac=args.tv_sh_sparsity,
                    ndc_coeffs=dset.ndc_coeffs,
                    contiguous=args.tv_contiguous,
                )
            if args.lambda_tv_lumisphere > 0.0:
                grid.inplace_tv_lumisphere_grad(
                    grid.sh_data.grad,
                    scaling=args.lambda_tv_lumisphere,
                    dir_factor=args.tv_lumisphere_dir_factor,
                    sparse_frac=args.tv_lumisphere_sparsity,
                    ndc_coeffs=dset.ndc_coeffs,
                )
            if args.lambda_l2_sh > 0.0:
                grid.inplace_l2_color_grad(grid.sh_data.grad, scaling=args.lambda_l2_sh)
            if grid.use_background and (args.lambda_tv_background_sigma > 0.0 or args.lambda_tv_background_color > 0.0):
                grid.inplace_tv_background_grad(
                    grid.background_data.grad,
                    scaling=args.lambda_tv_background_color,
                    scaling_density=args.lambda_tv_background_sigma,
                    sparse_frac=args.tv_background_sparsity,
                    contiguous=args.tv_contiguous,
                )
            if args.lambda_tv_basis > 0.0:
                tv_basis = grid.tv_basis()
                loss_tv_basis = tv_basis * args.lambda_tv_basis
                loss_tv_basis.backward()

            # Manual SGD/rmsprop step
            # ic(lr_sigma)
            if lr_sigma > 0.0:
                grid.optim_density_step(lr_sigma, beta=args.rms_beta, optim=args.sigma_optim)
            grid.optim_sh_step(lr_sh, beta=args.rms_beta, optim=args.sh_optim)

            if grid.use_background:
                grid.optim_background_step(lr_sigma_bg, lr_color_bg, beta=args.rms_beta, optim=args.bg_optim)

            if grid.basis_type == svox2.BASIS_TYPE_3D_TEXTURE:
                grid.optim_basis_step(lr_basis, beta=args.rms_beta, optim=args.basis_optim)
            elif grid.basis_type == svox2.BASIS_TYPE_MLP:
                optim_basis_mlp.step()
                optim_basis_mlp.zero_grad()


    img_id = np.random.randint(low=0, high=dset.n_images)
    cam = svox2.Camera(
        dset.c2w[img_id].to(device=device),
        dset.intrins.get("fx", img_id),
        dset.intrins.get("fy", img_id),
        dset.intrins.get("cx", img_id),
        dset.intrins.get("cy", img_id),
        width=dset.get_image_size(img_id)[1],
        height=dset.get_image_size(img_id)[0],
        ndc_coeffs=dset.ndc_coeffs,
    )
    rgb_pred = grid.volume_render_image(cam, use_kernel=True).detach().cpu().numpy()
    imageio.imwrite(
        os.path.join(args.train_dir, f"logim_{epoch_id}.png"),
        np.clip(rgb_pred * 255.0, 0.0, 255.0).astype(np.uint8),
    )

    if epoch_id < args.mse_num_epoches:
        epoch_size = dset.rays.origins.size(0)
        batch_size = args.batch_size
        batches_per_epoch = (epoch_size - 1) // batch_size + 1
        train_step(optim_type="ray")
    elif epoch_id < args.mse_num_epoches + args.nnfm_num_epoches:
        epoch_size = dset.n_images
        batch_size = 1
        batches_per_epoch = (dset.n_images - 1) // batch_size + 1
        train_step(optim_type="image")

    epoch_id += 1
    gstep_id_base += batches_per_epoch
    torch.cuda.empty_cache()
    gc.collect()

    if epoch_id >= args.mse_num_epoches + args.nnfm_num_epoches:
        if not args.no_post_ct:
            num_views, view_height, view_width = dset.n_images, dset.h, dset.w
            rgb_gt = dset.rays.gt.view(num_views, view_height, view_width, 3)
            with torch.no_grad():
                for img_id in range(num_views):
                    cam = svox2.Camera(
                        dset.c2w[img_id].to(device=device),
                        dset.intrins.get("fx", img_id),
                        dset.intrins.get("fy", img_id),
                        dset.intrins.get("cx", img_id),
                        dset.intrins.get("cy", img_id),
                        width=view_width,
                        height=view_height,
                        ndc_coeffs=dset.ndc_coeffs,
                    )
                    rgb_pred = grid.volume_render_image(cam, use_kernel=True)
                    rgb_gt[img_id] = rgb_pred.reshape(view_height, view_width, 3).contiguous().cpu().clamp_(0.0, 1.0)
            dset.rays.gt, color_tf = match_colors_for_image_set(dset.rays.gt, style_img)
            grid.apply_ct(color_tf.detach().cpu().numpy())

        global_stop_time = datetime.now()
        secs = (global_stop_time - global_start_time).total_seconds()
        timings_file = open(os.path.join(args.train_dir, "time_mins.txt"), "w")
        timings_file.write(f"{secs / 60}\n")
        timings_file.close()

        ckpt_path = path.join(args.train_dir, "ckpt.npz")
        grid.save(ckpt_path)

        img_id = np.random.randint(low=0, high=dset.n_images)
        cam = svox2.Camera(
            dset.c2w[img_id].to(device=device),
            dset.intrins.get("fx", img_id),
            dset.intrins.get("fy", img_id),
            dset.intrins.get("cx", img_id),
            dset.intrins.get("cy", img_id),
            width=dset.get_image_size(img_id)[1],
            height=dset.get_image_size(img_id)[0],
            ndc_coeffs=dset.ndc_coeffs,
        )
        rgb_pred = grid.volume_render_image(cam, use_kernel=True)
        rgb_pred = rgb_pred.detach().cpu().numpy()

        imageio.imwrite(
            os.path.join(args.train_dir, f"logim_{epoch_id}_final.png"),
            np.clip(rgb_pred * 255.0, 0.0, 255.0).astype(np.uint8),
        )

        break
