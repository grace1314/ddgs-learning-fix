import os
import sys
import math
from random import randint
from argparse import ArgumentParser
import numpy as np
import torch
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim
from utils.general_utils import safe_state


def compute_keep_mask(gaussians: GaussianModel,
                      scene_extent: float,
                      max_needle_ratio: float,
                      max_ws_scale_ratio: float,
                      min_opacity: float,
                      knn_std_mult: float,
                      roi_percentile: float) -> torch.Tensor:
    with torch.no_grad():
        xyz = gaussians.get_xyz
        scales = gaussians.get_scaling
        opacities = gaussians.get_opacity.squeeze(-1)

        scales_sorted, _ = torch.sort(scales, dim=1)
        s_min, s_mid, s_max = scales_sorted[:, 0], scales_sorted[:, 1], scales_sorted[:, 2]
        needle_ratio = s_max / (s_mid + 1e-7)

        keep = torch.ones(xyz.shape[0], dtype=torch.bool, device=xyz.device)

        if max_needle_ratio > 0.0:
            c_needle = needle_ratio <= max_needle_ratio
            print(f"  [Filter] Needle ratio > {max_needle_ratio}: {(~c_needle).sum().item()} removed")
            keep &= c_needle

        if max_ws_scale_ratio > 0.0:
            max_allowed = max_ws_scale_ratio * scene_extent
            c_scale = s_max <= max_allowed
            print(f"  [Filter] Max scale > {max_allowed:.4f}: {(~c_scale).sum().item()} removed")
            keep &= c_scale

        if min_opacity > 0.0:
            c_opac = opacities >= min_opacity
            print(f"  [Filter] Opacity < {min_opacity}: {(~c_opac).sum().item()} removed")
            keep &= c_opac

        if roi_percentile > 0.0:
            lo = torch.quantile(xyz, roi_percentile / 100.0, dim=0)
            hi = torch.quantile(xyz, 1.0 - roi_percentile / 100.0, dim=0)
            c_roi = ((xyz >= lo) & (xyz <= hi)).all(dim=1)
            print(f"  [Filter] Outside [{roi_percentile}%, {100.0 - roi_percentile}%] 3D ROI: {(~c_roi).sum().item()} removed")
            keep &= c_roi

        if knn_std_mult > 0.0 and xyz.shape[0] > 16:
            xyz_np = xyz.detach().cpu().numpy()
            nbrs = NearestNeighbors(n_neighbors=8, n_jobs=-1).fit(xyz_np)
            mean_d = nbrs.kneighbors()[0][:, 1:].mean(axis=1)
            thresh = mean_d.mean() + knn_std_mult * mean_d.std()
            c_knn = torch.from_numpy(mean_d <= thresh).to(xyz.device)
            print(f"  [Filter] Isolated 3D spatial outliers (KNN > {thresh:.4f}): {(~c_knn).sum().item()} removed")
            keep &= c_knn

    return keep


def clamp_gaussian_scales(gaussians: GaussianModel,
                          scene_extent: float,
                          max_needle_ratio: float,
                          max_ws_scale_ratio: float):
    with torch.no_grad():
        scales = gaussians.get_scaling
        scales_sorted, _ = torch.sort(scales, dim=1)
        s_mid = scales_sorted[:, 1:2]

        upper = torch.full_like(scales, float("inf"))
        if max_needle_ratio > 0.0:
            upper = torch.minimum(upper, s_mid * max_needle_ratio)
        if max_ws_scale_ratio > 0.0:
            upper = torch.minimum(upper, torch.full_like(scales, max_ws_scale_ratio * scene_extent))

        clamped = torch.minimum(scales, upper).clamp_min(1e-7)
        gaussians._scaling.copy_(gaussians.scaling_inverse_activation(clamped))


def clean_and_finetune(dataset, opt, pipe, iteration, max_needle_ratio, max_ws_scale_ratio,
                       min_opacity, knn_std_mult, roi_percentile, finetune_iters, save_iter):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=True)
    n_before = gaussians.get_xyz.shape[0]
    print(f"Loaded {n_before} Gaussians from iteration {scene.loaded_iter} (extent={scene.cameras_extent:.4f})")

    gaussians.training_setup(opt)
    keep_mask = compute_keep_mask(
        gaussians,
        scene.cameras_extent,
        max_needle_ratio=max_needle_ratio,
        max_ws_scale_ratio=max_ws_scale_ratio,
        min_opacity=min_opacity,
        knn_std_mult=knn_std_mult,
        roi_percentile=roi_percentile,
    )
    gaussians.prune_points(~keep_mask)
    clamp_gaussian_scales(gaussians, scene.cameras_extent, max_needle_ratio, max_ws_scale_ratio)
    n_after = gaussians.get_xyz.shape[0]
    print(f"Remaining Gaussians after geometric pruning: {n_after} ({100.0 * n_after / max(1, n_before):.1f}%)")

    if finetune_iters > 0:
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        viewpoint_stack = None

        # Freeze xyz movement slightly for stability during post-clean healing
        for pg in gaussians.optimizer.param_groups:
            if pg["name"] == "xyz":
                pg["lr"] = opt.position_lr_final * scene.cameras_extent

        pbar = tqdm(range(1, finetune_iters + 1), desc="Post-clean healing finetune")
        for step in pbar:
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
            gt_image = viewpoint_cam.original_image.cuda()

            render_pkg = render(viewpoint_cam, gaussians, pipe, background, is_train=False)
            image = render_pkg["render"]

            loss = l1_loss(image, gt_image) + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            if max_needle_ratio > 0.0:
                log_scales_sorted, _ = torch.sort(gaussians._scaling, dim=1)
                log_s_mid, log_s_max = log_scales_sorted[:, 1], log_scales_sorted[:, 2]
                loss = loss + 0.05 * torch.clamp(log_s_max - log_s_mid.detach() - math.log(max_needle_ratio * 0.95), min=0.0).mean()

            loss.backward()
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)
            clamp_gaussian_scales(gaussians, scene.cameras_extent, max_needle_ratio, max_ws_scale_ratio)

            if step % 50 == 0:
                pbar.set_postfix({"Loss": f"{loss.item():.5f}"})

    out_iter = save_iter if save_iter > 0 else (scene.loaded_iter + finetune_iters)
    scene.save(out_iter)
    clean_alias = os.path.join(dataset.model_path, "point_cloud_clean.ply")
    gaussians.save_ply(clean_alias)
    print(f"Saved cleaned model to iteration_{out_iter}/point_cloud.ply and {clean_alias}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Post-training Gaussian geometric cleaner and short healing finetuner")
    model = ModelParams(parser, sentinel=True)
    opt = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--clean_needle_ratio", default=8.0, type=float)
    parser.add_argument("--clean_ws_scale_ratio", default=0.0, type=float)
    parser.add_argument("--clean_min_opacity", default=0.02, type=float)
    parser.add_argument("--knn_std_mult", default=0.0, type=float)
    parser.add_argument("--roi_percentile", default=0.0, type=float)
    parser.add_argument("--finetune_iters", default=1000, type=int)
    parser.add_argument("--save_iter", default=0, type=int)
    parser.add_argument("--quiet", action="store_true")

    args = get_combined_args(parser)
    safe_state(args.quiet)
    clean_and_finetune(
        model.extract(args),
        opt.extract(args),
        pipeline.extract(args),
        iteration=args.iteration,
        max_needle_ratio=args.clean_needle_ratio,
        max_ws_scale_ratio=args.clean_ws_scale_ratio,
        min_opacity=args.clean_min_opacity,
        knn_std_mult=args.knn_std_mult,
        roi_percentile=args.roi_percentile,
        finetune_iters=args.finetune_iters,
        save_iter=args.save_iter,
    )
