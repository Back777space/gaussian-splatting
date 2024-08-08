
import os

model_path = "C:\\Users\\Jobstudent\\Desktop\\werk\\trained_models\\models"
image_path = "C:\\Users\\Jobstudent\\Desktop\\werk\\training_images"

models = ["train", "truck", "treehill", "flowers"]
# , "room", "bonsai"

for m in models:
    os.system(f"python render.py -m {model_path}\\{m} -s {image_path}\\{m} --progressive")
    os.system(f"python metrics.py -m {model_path}\\{m}")
    os.system(f"python progressive\\plot_metrics.py -m {model_path}\\{m}")
