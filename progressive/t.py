import torch

def unique_keep_order(t):
    _, inverse_indices = torch.unique(t, return_inverse=True)
    first_occurrence_mask = torch.zeros_like(t, dtype=torch.bool)
    first_occurrence_mask.scatter_(0, inverse_indices, torch.ones_like(inverse_indices, dtype=torch.bool))
    unique_in_order_tensor = t[first_occurrence_mask]
    return unique_in_order_tensor


t_ = torch.tensor([5,1,2,3,4,1,7,2,6,10,7,0,9,10,6])

w = unique_keep_order(t_)
print(w)
exit()