from pathlib import Path
import json
from argparse import ArgumentParser
import matplotlib.pyplot as plt

r = range(10,110,10)

def evaluate(model_paths):
    methods = ["antimatter", "30vc13contr57scale", "voxels_contr_antimatter_65", "frustum_contr"]

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

        plt.figure()
        plt.title("PSNR")
        plt.xlabel("percentage of splats rendered")
        plt.ylabel("psnr")
        for key, value in per_method.items():
            plt.plot(r, value["psnr"], label=key)
        plt.legend() 
        plt.savefig(f"{test_dir}\\psnr.png")

        plt.figure()
        plt.title("SSIM")
        plt.xlabel("percentage of splats rendered")
        plt.ylabel("ssim")
        for key, value in per_method.items():
            plt.plot(r, value["ssim"], label=key)
        plt.legend() 
        plt.savefig(f"{test_dir}\\ssim.png")

        plt.figure()
        plt.title("LPIPS")
        plt.xlabel("percentage of splats rendered")
        plt.ylabel("lpips")
        for key, value in per_method.items():
            plt.plot(r, value["lpips"], label=key)
        plt.legend() 
        plt.savefig(f"{test_dir}\\lpips.png")

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)
