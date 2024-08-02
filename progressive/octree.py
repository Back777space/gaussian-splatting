import open3d as o3d
from progressive.util import intersect_tensors_no_loop, reverse_mask
from progressive.weights import weigh_antimatter, weigh_contr_vc, weigh_contrib, weigh_vc_antimatter, weigh_vc_scale
from scene.gaussian_model import GaussianModel
import torch

LARGE_LEAF_THRESHOLD = 0
MAX_TREE_DEPTH = 3
MAIN_STRAT_PERCENTAGE = 0.65

def build_octree(gaussians: GaussianModel):
    print("Octree build started")
    octree = o3d.geometry.Octree(max_depth=MAX_TREE_DEPTH)
    octree.convert_from_point_cloud(build_point_cloud(gaussians))
    print("Octree build done")
    # o3d.visualization.draw_geometries([octree])
    return octree


def build_point_cloud(gaussians: GaussianModel):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(gaussians._xyz.detach().cpu().numpy())
    return pcd


def traverse_for_indices(node, p, indices, depth, weight_cb, frustum=None):
    if isinstance(node, o3d.geometry.OctreeLeafNode):
        node_indices_t = torch.tensor(node.indices, device='cuda:0')
        if frustum is not None:
            node_indices_t = intersect_tensors_no_loop(node_indices_t, frustum)

        total_points_amt = node_indices_t.numel()
        points_rendered = int(total_points_amt * p)  
        if points_rendered == 0:
            return indices
        
        points_main_strat = int(points_rendered * MAIN_STRAT_PERCENTAGE)
        points_scnd_strat = points_rendered - points_main_strat 

        _, largest_k_indices_main = weight_cb(node_indices_t, weigh_contrib, points_main_strat)

        # remove splats that have been chosen by the main strat so we dont select them again with the secondary one
        filtered_t = reverse_mask(node_indices_t, largest_k_indices_main)

        _, largest_k_indices_scnd = weight_cb(filtered_t, weigh_antimatter, points_scnd_strat)

        if points_main_strat == 0:
            indices = torch.cat((indices, filtered_t[largest_k_indices_scnd]), dim=0)
        elif points_scnd_strat == 0:
            indices = torch.cat((indices, node_indices_t[largest_k_indices_main]), dim=0)
        else:
            indices = torch.cat((indices, node_indices_t[largest_k_indices_main], filtered_t[largest_k_indices_scnd]), dim=0)
        return indices

    for idx in range(8):
        child_node = node.children[idx]
        if child_node is not None:
            indices = traverse_for_indices(child_node, p, indices, depth + 1, weight_cb, frustum)
    return indices


def count_leaves(node, count):
    if isinstance(node, o3d.geometry.OctreeLeafNode):
        return count + 1

    for idx in range(8):
        child_node = node.children[idx]
        if child_node is not None:
            count = count_leaves(child_node, count)
    return count
