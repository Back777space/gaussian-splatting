import torch
from tqdm import tqdm

from gaussian_renderer import render
from scene.gaussian_model import GaussianModel

class GaussianData:
    def __init__(self, gaussians: GaussianModel):
        amount = gaussians.get_opacity.shape[0]
        self.opacity = torch.squeeze(gaussians.get_opacity)
        self.scale = torch.squeeze(gaussians.get_scaling)
        self.viewcount = torch.squeeze(torch.zeros(amount, 1, dtype=torch.int32)).cuda()
        self.contrib = torch.squeeze(torch.zeros(amount, 1, dtype=torch.float32)).cuda()
        self.rgbs = torch.squeeze(torch.zeros(amount, 3, dtype=torch.float32)).cuda()

    def __eq__(self, other) : 
        return (
            self.opacity == other.opacity &
            self.scale == other.scale & 
            self.viewcount == other.viewcount & 
            self.contrib == other.contrib & 
            self.rgbs == other.rgbs
        )
    
    def set_ids_and_contr(self, gaussians, views, pipeline, background):
        for idx, view in enumerate(tqdm(views, desc="Pre-processing Gaussians")):
            output = render(view, gaussians, pipeline, background)
            ids_per_pixel = output["ids_per_pixel"]
            contr_per_pixel = output["contr_per_pixel"]
            self.update(
                ids_per_pixel,
                contr_per_pixel,
            )
            # save_mask(view, contr_per_pixel, os.path.join(model_path, name), '{0:05d}'.format(idx))
            # gt = view.original_image[0:3, :, :]
            # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

    def update(
        self,
        ids_per_pixel: torch.Tensor,
        contr_per_pixel: torch.Tensor,
    ):
        # set viewcount
        set = torch.unique(ids_per_pixel)
        self.viewcount.index_add_(0, set, torch.ones_like(set, dtype=torch.int32))

        # set total contribution
        self.contrib.index_add_(0, ids_per_pixel, contr_per_pixel)

        # dit werkt voor de kloten
        # self.viewcount[set] += 1
        # self.contrib[ids_per_pixel] += contr_per_pixel

    @staticmethod
    def mask_gaussians(gaussians: GaussianModel, mask):
        gaussians._opacity = gaussians._opacity[mask]
        gaussians._xyz = gaussians._xyz[mask]
        gaussians._scaling = gaussians._scaling[mask]
        gaussians._features_dc = gaussians._features_dc[mask]
        gaussians._features_rest = gaussians._features_rest[mask]
        gaussians._rotation = gaussians._rotation[mask]

