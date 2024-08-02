#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from progressive.gaussian_data import GaussianData
from progressive.octree import build_octree, traverse_for_indices
import torch
from progressive.weights import *
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from scene.cameras import Camera
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import copy
import time
import numpy as np

# TODO: refactor, test andere scenes

def render_set(
        model_path,
        name,
        iteration,
        views,
        gaussians: GaussianModel,
        pipeline,
        background,
    ):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        output = render(view, gaussians, pipeline, background)
        rendering = output["render"]

        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_progressive(
        model_path,
        name,
        iteration,
        views,
        gaussians,
        background,
    ):
    # octree = build_octree(gaussians)
    gaussian_data = GaussianData(gaussians)
    gaussian_amt = gaussians._opacity.shape[0]

    method_path = os.path.join(model_path, name, "30vc13contr57scale")
    makedirs(method_path, exist_ok=True)
    gts_path = os.path.join(method_path, "gt")
    makedirs(gts_path, exist_ok=True)

    # # s = time.time()
    # in_frustum_mask = frustum_check(gaussians, views[0])
    # in_frustum_indices = torch.nonzero(in_frustum_mask, as_tuple=False).squeeze()
    # # e = time.time()
    # # print((e - s) * 1000, "ms")
    # frustum_gaussians = copy.deepcopy(gaussians)
    # GaussianData.mask_gaussians(frustum_gaussians, in_frustum_indices)
    
    # set up gaussian data
    for idx, view in enumerate(tqdm(views, desc="Pre-processing Gaussians")):
        output = render(view, gaussians, pipeline, background)
        ids_per_pixel = output["ids_per_pixel"]
        contr_per_pixel = output["contr_per_pixel"]
        gaussian_data.update(
            ids_per_pixel,
            contr_per_pixel,
        )
        # gt = view.original_image[0:3, :, :]
        # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

    # render progressive [0.1 - 1]
    for i in range(10, 100, 10):
        p = i / 100
        progressive_path = os.path.join(method_path, "progressive_{}_{}".format(iteration, p))
        makedirs(progressive_path, exist_ok=True)
        
        gaussians_c = copy.deepcopy(gaussians)
        order = get_indices_to_render(
            gaussian_data,
            gaussian_amt,
            p, frustum=None, octree=None
        )
        GaussianData.mask_gaussians(gaussians_c, order)

        for idx, view in enumerate(tqdm(views, desc=f"Progressive loading {i}%")):
            # ------------ FRUSTUM CODE ------------
            # rendered_in_frustum = min(p, in_frustum_indices.numel() / gaussian_amt)
            # rendered_out_frustum = p - rendered_in_frustum

            # indices_f = get_indices_to_render(
            #     gaussian_data, 
            #     gaussian_amt,
            #     rendered_in_frustum, frustum=in_frustum_indices
            # )
            # indices = in_frustum_indices[indices_f]

            # if rendered_out_frustum > 0:
            #     out_frustum_indices = torch.nonzero(in_frustum_mask == False, as_tuple=False).squeeze()
            #     most_contributing_out_frustum = get_indices_to_render(
            #         gaussian_data, 
            #         gaussian_amt,
            #         rendered_out_frustum, frustum=out_frustum_indices
            #     )
            #     indices = torch.cat((indices, out_frustum_indices[most_contributing_out_frustum]), dim=0)
            
            # gcpy = copy.deepcopy(gaussians)
            # GaussianData.mask_gaussians(gcpy, indices)
            # ------------------------------------------

            rendering = render(view, gaussians_c, pipeline, background)["render"]
            torchvision.utils.save_image(rendering, os.path.join(progressive_path, '{0:05d}'.format(idx) + ".png"))


def get_indices_to_render(
        gaussian_data: GaussianData,
        total_amt_gaussians: int,
        percentage: float,
        octree=None, frustum=None
    ):
    if octree is not None:
        return indices_octree(
            octree, percentage, 
            lambda mask, weigh_f, amt: torch.topk(
                weigh_f(gaussian_data, mask),
                amt,
                largest=True
            ), frustum
        )
    else:
        weights = weigh_gaussians_13contr_57scale_30vc(gaussian_data)
        largest_k_values, largest_k_indices = torch.topk(
            weights,
            int(total_amt_gaussians * percentage),
            largest=True
        )
        return largest_k_indices


def avg_col(rgbs, largest_indices):
    # rgbs[largest_indices]
    return torch.mean(rgbs).item()

def no_rest(gaussians: GaussianModel, original=None, exclude_indices=None):
    gaussians._features_rest = torch.zeros_like(gaussians._features_rest).cuda()
    if exclude_indices is not None:
        gaussians._features_rest[exclude_indices] = original._features_rest[exclude_indices]


def set_col(gaussians: GaussianModel, col, exclude_indices, original: GaussianModel, rest=True):
    gaussians._features_dc = torch.full(gaussians._features_dc.shape, col).cuda()
    gaussians._features_dc[exclude_indices] = original._features_dc[exclude_indices]
    if not rest:
        no_rest(gaussians, original, exclude_indices)

def indices_octree(octree, p, weight_cb, frustum=None):
    indices = torch.empty(0, dtype=torch.int32, device='cuda:0')
    indices = traverse_for_indices(
        octree.root_node, p, indices,
        0, weight_cb, frustum
    )
    return indices

# returns size in bytes
def get_size(gaussians: GaussianModel):
    dc_coeffs_size = gaussians._features_dc.element_size() * (gaussians._features_dc != 0).all(dim=1).sum().item()
    rest_coeffs = gaussians._features_rest.element_size() * (gaussians._features_rest != 0).all(dim=1).sum().item()
    return (dc_coeffs_size, rest_coeffs)


def write_sh_size(gaussians, model_path, name, p):
    size = get_size(gaussians)
    with open(os.path.join(model_path, name, "avgnorest_full") + ".txt", "a") as f:
        f.write(f"\n{p}\ndeg 0 coeffs: {size[0]/1000000} MB\n")
        f.write(f"rest coeffs: {size[1]/1000000} MB\n\n")

def frustum_check(gaussians: GaussianModel, cam: Camera):
    transform = cam.full_proj_transform
    positions = gaussians.get_xyz

    ones = torch.ones((positions.shape[0], 1), dtype=positions.dtype).cuda()
    positions_homo = torch.cat((positions, ones), dim=1)
    transformed = positions_homo @ transform
    pos = transformed[..., :3] / transformed[..., 3:]

    in_frustum = (
        (pos[..., 0] >= -1.5) & (pos[..., 0] <= 1.5) &
        (pos[..., 1] >= -1.5) & (pos[..., 1] <= 1.5) &
        (pos[..., 2] <= 1) 
    )
    return in_frustum


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, prog: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if prog:
            render_progressive(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, background)
            return

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--progressive", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.progressive)