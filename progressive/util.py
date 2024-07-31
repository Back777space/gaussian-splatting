import torch

def intersection(t1, t2):
    t1 = t1.cuda(); t2 = t2.cuda();
    intersection_mask = torch.isin(t1, t2).cuda()
    intersection = t1[intersection_mask]
    return intersection