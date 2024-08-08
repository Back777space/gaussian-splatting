import json
import os
import torch
from scene.gaussian_model import GaussianModel

def intersect_tensors_no_loop(tensor1, tensor2):
    unique_tensor2 = torch.unique_consecutive(tensor2)
    mask = torch.isin(tensor1, unique_tensor2)
    intersection_tensor = tensor1[mask]
    return intersection_tensor

def unique_keep_order(t):
    _, inverse_indices = torch.unique(t, return_inverse=True)
    first_occurrence_mask = torch.zeros_like(t, dtype=torch.bool)
    first_occurrence_mask[torch.cumsum(first_occurrence_mask.index_add(0, inverse_indices, torch.ones_like(inverse_indices, dtype=torch.bool)), dim=0) == 1] = True
    unique_in_order_tensor = t[first_occurrence_mask]
    return unique_in_order_tensor

def reverse_mask(tensor, m):
    mask = torch.ones_like(tensor, dtype=torch.bool, device='cuda:0')
    mask[m] = False
    return tensor[mask]

def empty_tensor():
    return torch.empty(0, device='cuda:0', dtype=torch.long)

def write_sh_size(gaussians, model_path, name, p):
    size = get_size(gaussians)
    with open(os.path.join(model_path, name, "avgnorest_full") + ".txt", "a") as f:
        f.write(f"\n{p}\ndeg 0 coeffs: {size[0]/1000000} MB\n")
        f.write(f"rest coeffs: {size[1]/1000000} MB\n\n")

def write_timing(start, ends, model_path, name, fname):
    with open(os.path.join(model_path, name, f"timing_{fname}") + ".json", "w") as f:
        data = {start: ends}
        json.dump(data, f)

# returns size in bytes
def get_size(gaussians: GaussianModel):
    dc_coeffs_size = gaussians._features_dc.element_size() * (gaussians._features_dc != 0).all(dim=1).sum().item()
    rest_coeffs = gaussians._features_rest.element_size() * (gaussians._features_rest != 0).all(dim=1).sum().item()
    return (dc_coeffs_size, rest_coeffs)