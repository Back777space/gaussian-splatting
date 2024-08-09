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
from progressive.mask import save_mask
from progressive.octree import build_octree, traverse_for_indices
import torch
from progressive.util import empty_tensor
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
        frustum_culling=False,
        octree=None
    ):
    method_path = os.path.join(model_path, name, "contrib_depth_1")
    makedirs(method_path, exist_ok=True)
    gts_path = os.path.join(method_path, "gt")
    makedirs(gts_path, exist_ok=True)
    
    gaussian_data = GaussianData(gaussians)
    gaussian_data.set_ids_and_contr(gaussians, views, pipeline, background)
    gaussian_amt = gaussian_data.opacity.shape[0]

    if frustum_culling:
        in_frustum_mask = frustum_check(gaussians, views[0])
        in_frustum_indices = torch.nonzero(in_frustum_mask, as_tuple=False).squeeze()
        out_frustum_indices = torch.nonzero(in_frustum_mask == False, as_tuple=False).squeeze()

        indices_f = get_indices_to_render(
            gaussian_data, in_frustum_indices.shape[0]/gaussian_amt, 
            frustum=in_frustum_indices, octree=octree
        )
        order_in_frustum = in_frustum_indices[indices_f]
        indices_of = get_indices_to_render(
            gaussian_data, out_frustum_indices.shape[0]/gaussian_amt, 
            frustum=out_frustum_indices, octree=octree
        )
        order_out_frustum = out_frustum_indices[indices_of]
    else:
        order = get_indices_to_render(
            gaussian_data, 
            1.0,
            octree=octree
        )

    # render progressive [0.1 - 1]
    for i in range(10, 110, 10):
        p = i / 100
        progressive_path = os.path.join(method_path, "progressive_{}_{}".format(iteration, p))
        makedirs(progressive_path, exist_ok=True)

        rendered_gaussians = int(p * gaussian_amt)
        indices = None
        if frustum_culling:
            p_in_frustum = p * 0.9
            rendered_in_frustum = min(p_in_frustum, in_frustum_indices.shape[0] / gaussian_amt)
            rest = p_in_frustum - rendered_in_frustum
            rendered_out_frustum = max(rest, p - rendered_in_frustum)

            part1 = order_in_frustum[:int(gaussian_amt * rendered_in_frustum)]
            part2 = order_out_frustum[:int(gaussian_amt * rendered_out_frustum)]
            indices = torch.cat((part1, part2), dim=0)
        else:
            indices = order[:rendered_gaussians]

        gcpy = copy.deepcopy(gaussians)
        GaussianData.mask_gaussians(gcpy, indices)
        
        for idx, view in enumerate(tqdm(views, desc=f"Progressive loading {i}%")):
            rendering = render(view, gcpy, pipeline, background)["render"]
            torchvision.utils.save_image(rendering, os.path.join(progressive_path, '{0:05d}'.format(idx) + ".png"))
        
def get_indices_to_render(
        gaussian_data: GaussianData,
        percentage: float,
        octree=None, frustum=None
    ):
    order = None
    if octree is not None:
        order = indices_octree(
            octree, percentage, 
            lambda mask, weigh_f, amt: torch.topk(
                weigh_f(gaussian_data, mask),
                amt,
                largest=True
            ), frustum
        )
    else:
        weights = weigh_contrib(gaussian_data, frustum)
        _, order = torch.topk(
            weights,
            int(gaussian_data.opacity.shape[0] * percentage),
            largest=True
        )
    return order

def set_col(gaussians: GaussianModel, col, exclude_indices, original: GaussianModel):
    gaussians._features_dc = torch.full(gaussians._features_dc.shape, col).cuda()
    gaussians._features_dc[exclude_indices] = original._features_dc[exclude_indices]

def indices_octree(octree, p, weight_cb, frustum=None):
    indices = empty_tensor()
    step = 10

    # we can't just use indices_octree(1.0)[:percentage] because every gaussian of a voxel that was traversed first 
    # will be in front of the gaussians from other voxels, eventhough the others may have a higher importance
    for i in range(step, int(p*100)+step, step):
        p = i / 100
        t = empty_tensor()
        voxel_parts = traverse_for_indices(
            octree.root_node, p-(step/100), p, t,
            weight_cb, frustum
        )
        # exclude elements already in indices 
        # (avoid using torch.unique here because it messes up the order even when sorted=False)
        mask = ~torch.isin(voxel_parts, indices)
        filtered_t = voxel_parts[mask]
        indices = torch.cat((indices, filtered_t), dim=0)
    return indices

def frustum_check(gaussians: GaussianModel, cam: Camera):
    transform = cam.full_proj_transform
    positions = gaussians.get_xyz

    ones = torch.ones((positions.shape[0], 1), dtype=positions.dtype).cuda()
    positions_homo = torch.cat((positions, ones), dim=1)
    transformed = positions_homo @ transform

    # check z axis before screen space division
    in_frustum_z = transformed[..., 2] >= 0

    pos = transformed[..., :3] / transformed[..., 2:3]

    in_frustum = (
        (pos[..., 0] >= -1.5) & (pos[..., 0] <= 1.5) &
        (pos[..., 1] >= -1.5) & (pos[..., 1] <= 1.5) &
        in_frustum_z 
    )
    return in_frustum


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, prog: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if prog:
            octree = build_octree(gaussians)
            render_progressive(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), 
                               gaussians, background, frustum_culling=False, octree=octree)
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