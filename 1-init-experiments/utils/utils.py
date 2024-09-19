from typing import Optional
import torch

def pad3d(
    tensor: torch.Tensor,
    sizes: torch.LongTensor,
    max_size: Optional[int] = None
):
    assert tensor.ndim == 2
    assert tensor.size(0) == sizes.sum()
    offset = [0] + torch.Tensor.tolist(sizes.cumsum(0))
    max_n_nodes = sizes.max() if max_size is None else max_size
    padded_tensor = torch.zeros(sizes.size(0), max_n_nodes, tensor.size(1)).to(tensor.device)
    for i in range(sizes.size(0)):
        padded_tensor[i][:sizes[i]] = tensor[offset[i]:offset[i+1]]
    return padded_tensor

def edge_index2adj_with_weight(
    edge_index: torch.LongTensor,
    edge_weight: torch.Tensor,
    n_nodes: torch.LongTensor
):
    n_tot_nodes, n_max_nodes = n_nodes.sum(), n_nodes.max()
    adj_ = torch.zeros(n_tot_nodes, n_tot_nodes, dtype=torch.uint8).to(n_nodes.device)
    adj_[edge_index[0], edge_index[1]] = 1
    adj_weight_ = torch.zeros(n_tot_nodes, n_tot_nodes).to(n_nodes.device)
    adj_weight_[edge_index[0], edge_index[1]] = edge_weight

    offset = torch.cat([torch.zeros(1, dtype=torch.long).to(n_nodes.device), n_nodes.cumsum(0)])
    adj = torch.zeros(n_nodes.size(0), n_max_nodes, n_max_nodes, dtype=torch.uint8).to(n_nodes.device)
    adj_weight = torch.zeros(n_nodes.size(0), n_max_nodes, n_max_nodes).to(n_nodes.device)
    for i in range(n_nodes.size(0)):
        adj[i][:n_nodes[i], :n_nodes[i]] = adj_[offset[i]:offset[i + 1], offset[i]:offset[i + 1]]
        adj_weight[i][:n_nodes[i], :n_nodes[i]] = adj_weight_[offset[i]:offset[i + 1], offset[i]:offset[i + 1]]
    return adj, adj_weight