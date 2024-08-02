import torch

def intersect_tensors_no_loop(tensor1, tensor2):
    unique_tensor2 = torch.unique(tensor2)
    mask = torch.isin(tensor1, unique_tensor2)
    intersection_tensor = tensor1[mask]
    return intersection_tensor

def reverse_mask(tensor, m):
    mask = torch.ones_like(tensor, dtype=torch.bool, device='cuda:0')
    mask[m] = False
    return tensor[mask]