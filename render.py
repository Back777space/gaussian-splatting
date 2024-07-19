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

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
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
        percentage_loaded = 1
    ):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    progressive_path = os.path.join(model_path, name, "progressive_{}_{}_{}"
                                .format(iteration, percentage_loaded, "scalevc"))

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(progressive_path, exist_ok=True)

    # data for each gaussian: 
    # (
    #   amount of views it is in, 
    #   total contribution over all views,
    #   opacity,
    #   scale,
    # )
    gaussian_amt = gaussians.get_opacity.shape[0]
    gaussian_data = torch.zeros(gaussian_amt, 4, dtype=torch.float32)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        output = render(view, gaussians, pipeline, background)
        rendering = output["render"]
        ids_per_pixel = output["ids_per_pixel"]
        contr_per_pixel = output["contr_per_pixel"]
        update_gaussian_view_data(
            gaussian_data,
            ids_per_pixel,
            contr_per_pixel,
            gaussians
        )
        
        # gt = view.original_image[0:3, :, :]
        # torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

    weights = weigh_gaussians(gaussian_data)
    largest_k_values, largest_k_indices = torch.topk(
        weights,
        int(gaussian_amt * percentage_loaded),
        largest=True
    )
    partial = copy.deepcopy(gaussians)
    mask_gaussians(partial, largest_k_indices)
    for idx, view in enumerate(tqdm(views, desc="Progressive loading")):
        if idx < 8:
            continue
        
        rendering = render(view, partial, pipeline, background)["render"]
        torchvision.utils.save_image(rendering, os.path.join(progressive_path, '{0:05d}'.format(idx - 8) + ".png"))
        
        if idx > 12:
            break

def update_gaussian_view_data(
    gaussian_data: torch.Tensor,
    ids_per_pixel: torch.Tensor,
    contr_per_pixel: torch.Tensor,
    gaussians: GaussianModel,
):
    # set viewcount
    set = torch.unique(ids_per_pixel)
    set = set.cpu()
    gaussian_data[set, 0] += 1.0
    
    # set total contribution
    ids_per_pixel = ids_per_pixel.cpu()
    contr_per_pixel = contr_per_pixel.cpu()
    gaussian_data[ids_per_pixel, 1] += contr_per_pixel

    # set scale and opacity
    gaussian_data[:, 2] = gaussians._opacity[:, 0]
    gaussian_data[:, 3] = torch.prod(torch.exp(gaussians._scaling), dim=1)


def normalize(t: torch.Tensor):
    return (t - torch.min(t)) / (torch.max(t) - torch.min(t))


def weigh_gaussians(gaussian_data):
    normalized_opacity = normalize(gaussian_data[:,2])
    normalized_scale = normalize(gaussian_data[:,3])
    normalized_vc = normalize(gaussian_data[:,0])
    normalized_contr = normalize(gaussian_data[:,1])

    return 0.001 * normalized_vc + normalized_scale * 0.999

def mask_gaussians(gaussians, mask):
    gaussians._opacity = gaussians._opacity[mask] 
    gaussians._xyz = gaussians._xyz[mask]
    gaussians._scaling = gaussians._scaling[mask]
    gaussians._features_dc = gaussians._features_dc[mask]
    gaussians._features_rest = gaussians._features_rest[mask]
    gaussians._rotation = gaussians._rotation[mask]

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
            for i in range(10, 100, 10):
                render_set(
                    dataset.model_path,
                    "test", 
                    scene.loaded_iter, 
                    scene.getTestCameras(), 
                    gaussians, 
                    pipeline, 
                    background,
                    i / 100
                )

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)