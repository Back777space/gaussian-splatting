from pathlib import Path
import json
from argparse import ArgumentParser
import matplotlib.pyplot as plt

def make_plot(name: str, strats, dir):
    plt.figure()
    plt.title(name.upper())
    plt.xlabel("time in s")
    plt.ylabel("percentage of splats rendered")

    ps = [p for p in range(0,110,10)]
    for key, value in strats.items():
        plt.plot([0] + value, [0] + ps, label=key)
    
    plt.legend() 
    plt.show()
    plt.savefig(f"{dir}\\{name}.png")

def evaluate(model_paths):
    for scene_dir in model_paths:
        test_dir = Path(scene_dir) / "test"
        strats = dict()
        for file in ["timing_no_frustum.json", "timing_frustum.json"]:
            try:
                with open(test_dir / file) as fp:
                    data = json.load(fp)
                    start = list(data)[0]     
                    ends = data[start]   
                    start = round(float(start), 6)
                    ends = [round(float(e), 6) - start for e in ends]
                    strats[file] = ends
            except:
                print("failed at file", file)
        make_plot("timing", strats, test_dir)            


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)
