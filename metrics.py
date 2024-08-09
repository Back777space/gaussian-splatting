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

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr, psnr_masked
from argparse import ArgumentParser

def readImages(renders_dir, gt_dir, mask_dir=None):
    renders = []
    gts = []
    image_names = []
    masks = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)

        render_t = tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda()
        gt_t = tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda()
        renders.append(render_t)
        gts.append(gt_t)

        image_names.append(fname)

        if mask_dir is not None:
            mask = Image.open(mask_dir / fname)
            mask_t = tf.to_tensor(mask).unsqueeze(0)[:, :3, :, :].cuda()
            mask_t = (mask_t != 0).any(dim=1, keepdim=True)
            masks.append(mask_t)

    return renders, gts, image_names, masks

def evaluate(model_paths):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            # for method in os.listdir(test_dir):
            for method in ["contrib_depth_1", "contrib_depth_2", "contrib_depth_3", "contrib_depth_4", "contrib_depth_5"]:
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir / "progressive_30000_1.0"

                for i in range(10, 110, 10):
                    p = i / 100

                    renders, gts, image_names, masks = readImages(method_dir / f"progressive_30000_{p}", gt_dir, test_dir / "masks")

                    ssims = []
                    psnrs = []
                    lpipss = []

                    for idx in tqdm(range(len(renders)), desc=f"Metric evaluation progress {i}%"):
                        ssim_ = ssim(renders[idx], gts[idx])
                        psnr_ = torch.clamp(psnr(renders[idx], gts[idx]), max=100)
                        lpips_ = lpips(renders[idx], gts[idx], net_type='vgg')    
                        ssims.append(ssim_)
                        psnrs.append(psnr_)
                        lpipss.append(lpips_)

                        # mask = masks[idx].expand(1, 3, -1, -1)
                        # psnr_masked = torch.clamp(psnr_masked(renders[idx], gts[idx]), max=100)
                        # masked = psnr_masked[mask].mean()

                    print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                    print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                    print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                    print("")

                    full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                            "PSNR": torch.tensor(psnrs).mean().item(),
                                                            "LPIPS": torch.tensor(lpipss).mean().item()})
                    per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                                "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                                "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})
                    with open(f"{scene_dir}\\test\\{method}\\results_{p}.json", "w") as fp:
                        json.dump(full_dict[scene_dir], fp, indent=True)
                    with open(f"{scene_dir}\\test\\{method}\\per_view_{p}.json", "w") as fp:
                        json.dump(per_view_dict[scene_dir], fp, indent=True)

                    full_dict[scene_dir][method] = {}
                    per_view_dict[scene_dir][method] = {}

        except Exception as e:
            print("Unable to compute metrics for model", scene_dir)
            print(e)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)
