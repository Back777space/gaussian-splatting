import torch

from scene.gaussian_model import GaussianModel

class GaussianData:
    def __init__(self, gaussians: GaussianModel):
        amount = gaussians.get_opacity.shape[0]
        self.opacity = torch.squeeze(gaussians.get_opacity)
        self.scale = torch.squeeze(gaussians.get_scaling)
        self.viewcount = torch.squeeze(torch.zeros(amount, 1, dtype=torch.int32)).cuda()
        self.contrib = torch.squeeze(torch.zeros(amount, 1, dtype=torch.float32)).cuda()
        self.rgbs = torch.squeeze(torch.zeros(amount, 3, dtype=torch.float32)).cuda()

    def update(
        self,
        ids_per_pixel: torch.Tensor,
        contr_per_pixel: torch.Tensor,
    ):
        # set viewcount
        set = torch.unique(ids_per_pixel)
        set = set.cpu()
        self.viewcount[set] += 1

        # set total contribution
        self.contrib[ids_per_pixel] += contr_per_pixel

    @staticmethod
    def mask_gaussians(gaussians: GaussianModel, mask):
        gaussians._opacity = gaussians._opacity[mask]
        gaussians._xyz = gaussians._xyz[mask]
        gaussians._scaling = gaussians._scaling[mask]
        gaussians._features_dc = gaussians._features_dc[mask]
        gaussians._features_rest = gaussians._features_rest[mask]
        gaussians._rotation = gaussians._rotation[mask]
