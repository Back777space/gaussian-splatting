from pathlib import Path
import json
from argparse import ArgumentParser
import matplotlib.pyplot as plt

def make_plot(name: str, per_method, test_dir):
    plt.figure()
    plt.title(name.upper())
    plt.xlabel("percentage of splats rendered")
    plt.ylabel(name)
    for key, value in per_method.items():
        plt.plot(r, value[name], label=key)
    plt.legend() 
    plt.savefig(f"{test_dir}\\{name}.png")

r = range(10,100,10)

def evaluate(model_paths):
    methods = ["voxels_contr_antimatter_80_depth_3_fixed", "antimatter", "30vc13contr57scale", "voxels_contr_antimatter_65_depth_3"]

    for scene_dir in model_paths:
        print("Scene:", scene_dir)
        test_dir = Path(scene_dir) / "test"
        # train_dir = Path(scene_dir) / "train"
        per_method = dict()

        for method in methods:
            per_method[method] = dict()

            print("Method:", method)
            method_dir = test_dir / method

            ssims = []
            psnrs = []
            lpipss = []
            for p in r:
                p = p / 100
                with open(method_dir / f"results_{p}.json") as f:
                    data = json.load(f)[method]
                    ssims.append(data["SSIM"])
                    psnrs.append(data["PSNR"])
                    lpipss.append(data["LPIPS"])

            per_method[method]["psnr"] = psnrs
            per_method[method]["ssim"] = ssims
            per_method[method]["lpips"] = lpipss

        make_plot("psnr", per_method, test_dir)
        make_plot("ssim", per_method, test_dir)
        make_plot("lpips", per_method, test_dir)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)
