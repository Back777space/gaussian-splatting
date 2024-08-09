import open3d as o3d
from progressive.util import intersect_tensors_no_loop
from progressive.weights import weigh_contrib, weight_contrib_antimatter
from scene.gaussian_model import GaussianModel
import torch
from math import ceil

MAX_TREE_DEPTH = 1

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

def merge_indices(weights, t):
    weights = weights[t]
    sorted_indices = torch.argsort(weights, descending=True)
    merged = t[sorted_indices]
    return merged

def traverse_for_indices(node, pstart, pend, indices, weight_cb, frustum=None):
    if isinstance(node, o3d.geometry.OctreeLeafNode):
        node_indices_t = torch.tensor(node.indices, device='cuda:0')
        if frustum is not None:
            node_indices_t = intersect_tensors_no_loop(node_indices_t, frustum)

        total_points_amt = node_indices_t.shape[0]

        _, largest_k_indices_main = weight_cb(node_indices_t, weigh_contrib, node_indices_t.shape[0])
        largest_k_indices_main = largest_k_indices_main[int(total_points_amt * pstart) : ceil(total_points_amt * pend)]

        indices_main = node_indices_t[largest_k_indices_main]

        indices = torch.cat((indices, indices_main), dim=0)
        
        return indices

    for idx in range(8):
        child_node = node.children[idx]
        if child_node is not None:
            indices = traverse_for_indices(child_node, pstart, pend, indices, weight_cb, frustum)
    return indices


def count_leaves(node, count):
    if isinstance(node, o3d.geometry.OctreeLeafNode):
        return count + 1

    for idx in range(8):
        child_node = node.children[idx]
        if child_node is not None:
            count = count_leaves(child_node, count)
    return count
