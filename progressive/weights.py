import torch
from progressive.gaussian_data import GaussianData

def normalize(t: torch.Tensor):
    if t.numel() == 0:
        return t
    return (t - torch.min(t)) / (torch.max(t) - torch.min(t))

def weigh_vc(gaussian_data: GaussianData, mask):
    if mask is None:
        vc = gaussian_data.viewcount
    else:
        vc = gaussian_data.viewcount[mask]
    return normalize(vc)

def weigh_contrib(gaussian_data: GaussianData, mask):
    if mask is None:
        contrib = gaussian_data.contrib
    else:
        contrib = gaussian_data.contrib[mask]
    return normalize(contrib)

def weigh_opacity(gaussian_data: GaussianData, mask):
    if mask is None:
        opacity = gaussian_data.opacity
    else:
        opacity = gaussian_data.opacity[mask]
    return normalize(opacity)

def weigh_scale(gaussian_data: GaussianData, mask):
    if mask is None:
        scale = gaussian_data.scale
    else:
        scale = gaussian_data.scale[mask]
    return normalize(torch.sum(scale, 1))

def weigh_vc_antimatter(gaussian_data, mask=None):
    return 0.7 * weigh_antimatter(gaussian_data, mask) + 0.3 * weigh_vc(gaussian_data, mask)

def weigh_vc_scale(gaussian_data, mask=None):
    return 0.5 * weigh_scale(gaussian_data, mask) + 0.5 * weigh_vc(gaussian_data, mask)

def weigh_contr_vc(gaussian_data, mask=None):
    return 0.8 * weigh_contrib(gaussian_data, mask) + 0.2 * weigh_vc(gaussian_data, mask)

def weigh_gaussians_13contr_57scale_30vc(gaussian_data: GaussianData, mask=None):
    return 0.13 + weigh_contrib(gaussian_data, mask) + 0.57 * weigh_scale(gaussian_data, mask) + 0.30 * weigh_vc(gaussian_data, mask)

def weigh_antimatter(gaussian_data: GaussianData, mask=None):
    if mask is None:
        scale = gaussian_data.scale
        opacity = gaussian_data.opacity
    else:
        scale = gaussian_data.scale[mask]
        opacity = gaussian_data.opacity[mask]
    return torch.exp(torch.sum(scale, 1))  / (1 + torch.exp(-opacity))
