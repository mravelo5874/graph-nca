# * This script is (nearly) directly taken from gengala/egnca/egnn.py.
# * Please refer to the original script for more details:
# * https://github.com/gengala/egnca/blob/main/egnn.py

import torch
from typing import List

class EGCL(torch.nn.Module):
    def __init__(
        self,
        coord_dim: int, 
        node_dim: int, 
        message_dim: int,
        edge_attr_dim: int = 0,
        out_node_dim: int = None,
        has_vel: bool = False,
        normalize: bool = False,
        is_residual: bool = False,
        has_vel_norm: bool = False,
        has_attention: bool = False,
        has_coord_act: bool = False,
        act_name: str = 'silu',
        aggr_coord: str = 'mean',
        aggr_hidden:  str = 'sum',
    ):
        super(EGCL, self).__init__()
        assert aggr_coord == 'mean' or aggr_coord == 'sum'
        assert aggr_hidden == 'mean' or aggr_hidden == 'sum'
        
        self.coord_dim = coord_dim
        self.node_dim = node_dim
        self.message_dim = message_dim
        self.edge_attr_dim = edge_attr_dim
        self.out_node_dim = node_dim if out_node_dim is None else out_node_dim
        assert not is_residual or self.out_node_dim == node_dim, 'Skip connection allowed iff out_node_dim == node_dim'
        self.is_residual = is_residual
        self.has_attention = has_attention
        self.has_vel = has_vel
        self.has_vel_norm = has_vel_norm
        self.normalize = normalize
        self.aggr_coord = aggr_coord
        self.aggr_hidden = aggr_hidden
        self.has_coord_act = has_coord_act
        
        # * get activation function
        self.act_name = act_name
        assert act_name == 'tanh' or act_name == 'lrelu' or act_name == 'silu'
        act = {'tanh': torch.nn.Tanh(), 'lrelu': torch.nn.LeakyReLU(), 'silu': torch.nn.SiLU()}[act_name]
        
        # * create edge MLP
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(node_dim + node_dim + edge_attr_dim + 1, message_dim),
            act,
            torch.nn.Linear(message_dim, message_dim),
            act
        )

        # * create node MLP
        self.node_mlp = torch.nn.Sequential(
            torch.nn.Linear(message_dim + node_dim, message_dim),
            act,
            torch.nn.Linear(message_dim, self.out_node_dim)
        )

        # * create coord MLP
        last_coord_layer = torch.nn.Linear(message_dim, 1, bias=False)
        last_coord_layer.weight.data.zero_()
        self.coord_mlp = torch.nn.Sequential(
            torch.nn.Linear(message_dim, message_dim),
            act,
            last_coord_layer,
            torch.nn.Tanh() if has_coord_act else torch.nn.Identity()
        )

        # * create attention MLP
        if has_attention:
            self.attention_mlp = torch.nn.Sequential(
                torch.nn.Linear(message_dim, 1),
                torch.nn.Sigmoid()
            )
            
        # * create velocity MLP
        if has_vel:
            self.vel_mlp = torch.nn.Sequential(
                torch.nn.Linear(node_dim + 1 if has_vel_norm else node_dim, node_dim // 2),
                act,
                torch.nn.Linear(node_dim // 2, 1),
            )
         
    def __repr__(self):
        return f'{self.__class__.__name__} [coord-dim={self.coord_dim}, node-dim={self.node_dim}, message-dim={self.message_dim}, edge-attr-dim={self.edge_attr_dim}, act={self.act_name}, res={self.is_residual}, vel={self.has_vel}, attention={self.has_attention}]'
    
    def aggregated_sum(
        data: torch.Tensor,
        index: torch.LongTensor,
        num_segments: int,
        mean: bool = False
    ):
        index = index.unsqueeze(1).repeat(1, data.size(1))
        agg = data.new_full((num_segments, data.size(1)), 0).scatter_add_(0, index, data)
        if mean:
            counts = data.new_full((num_segments, data.size(1)), 0).scatter_add_(0, index, torch.ones_like(data))
            agg = agg / counts.clamp(min=1)
        return agg

    def n_nodes2mask(
        n_nodes: torch.LongTensor
    ):
        max_n_nodes = n_nodes.max()
        mask = torch.cat(
            [torch.cat([n_nodes.new_ones(1, n), n_nodes.new_zeros(1, max_n_nodes - n)], dim=1) for n in n_nodes], dim=0
        ).bool()
        return mask

    # * runs edge MLP model
    def edge_model(
        self,
        node_feat: torch.Tensor,
        edge_index: torch.LongTensor,
        coord_radial: torch.Tensor,
        edge_weight: torch.Tensor = None,
        edge_attr: torch.Tensor = None,
    ):
        # * choose between sparse and dense versions
        # * SPARSE VERSION:
        if node_feat.ndim == 2:
            if edge_attr is not None:
                assert edge_attr.size(1) == self.edge_attr_dim
                edge_feat = torch.cat([node_feat[edge_index[0]], node_feat[edge_index[1]], coord_radial, edge_attr], dim=1)
            else:
                edge_feat = torch.cat([node_feat[edge_index[0]], node_feat[edge_index[1]], coord_radial], dim=1)
            out = self.edge_mlp(edge_feat)
            if edge_weight is not None:
                out = edge_weight.unsqueeze(1) * out
            if self.has_attention:
                out = self.attention_mlp(out) * out
                
        # * DENSE VERSION:      
        else:
            node_feat_exp = node_feat.unsqueeze(2).expand(-1, -1, node_feat.size(1), -1)
            edge_feat = torch.cat([node_feat_exp, node_feat_exp.permute(0, 2, 1, 3)], dim=-1)
            if edge_attr is not None:
                assert edge_attr.size(-1) == self.edge_attr_dim
                edge_feat = torch.cat([edge_feat, coord_radial.unsqueeze(-1), edge_attr], dim=-1)
            else:
                edge_feat = torch.cat([edge_feat, coord_radial.unsqueeze(-1)], dim=-1)
            out = self.edge_mlp(edge_feat) * edge_index.unsqueeze(-1)
            if edge_weight is not None:
                out = edge_weight.unsqueeze(-1) * out
            if self.has_attention:
                out = self.attention_mlp(out) * out
        
        # * return out tensor
        return out
    
    # * runs coord MLP model
    def coord_model(
        self,
        coord: torch.Tensor,
        coord_diff: torch.Tensor,
        edge_feat: torch.Tensor,
        edge_index: torch.LongTensor,
        node_feat: torch.Tensor = None,
        vel: torch.Tensor = None
    ):
        # * choose between sparse and dense versions
        # * SPARSE VERSION:
        if coord.ndim == 2:
            trans = coord_diff * self.coord_mlp(edge_feat)
            coord_agg = self.aggregated_sum(trans, edge_index[0], coord.size(0), mean=self.aggr_coord == 'mean')
            if self.has_vel:
                if self.has_vel_norm:
                    vel_scale = self.vel_mlp(torch.cat([node_feat, torch.norm(vel, p=2, dim=-1, keepdim=True)], dim=-1))
                else:
                    vel_scale = self.vel_mlp(node_feat)
                vel = vel_scale * vel + coord_agg
                coord = coord + vel
                return coord, vel
            else:
                out = coord + coord_agg
                
        # * DENSE VERSION:      
        else:
            trans = coord_diff * self.coord_mlp(edge_feat)
            coord_agg = torch.sum(trans * edge_index.unsqueeze(-1), dim=2)
            if self.aggr_coord == 'mean':
                coord_agg = coord_agg / edge_index.sum(dim=-1, keepdim=True).clamp(min=1)
            if self.has_vel:
                if self.has_vel_norm:
                    vel_scale = self.vel_mlp(torch.cat([node_feat, torch.norm(vel, p=2, dim=-1, keepdim=True)], dim=-1))
                else:
                    vel_scale = self.vel_mlp(node_feat)
                vel = vel_scale * vel + coord_agg
                coord = coord + vel
                return coord, vel
            else:
                out = coord + coord_agg
            
        # * return out tensor
        return out
    
    # * runs node MLP model
    def node_model(
        self,
        node_feat: torch.Tensor,
        edge_feat: torch.Tensor,
        edge_index: torch.LongTensor,
        n_nodes: torch.LongTensor = None,
    ):
        # * choose between sparse and dense versions
        # * SPARSE VERSION:
        if node_feat.ndim == 2:
            edge_feat_agg = self.aggregated_sum(edge_feat, edge_index[0], node_feat.size(0), mean=self.aggr_hidden == 'mean')
            out = self.node_mlp(torch.cat([node_feat, edge_feat_agg], dim=1))
            if self.is_residual:
                out = node_feat + out
            
        # * DENSE VERSION:
        else:
            edge_feat_agg = torch.sum(edge_feat * edge_index.unsqueeze(-1), dim=2)
            if self.aggr_hidden == 'mean':
                edge_feat_agg = edge_feat_agg / edge_index.sum(dim=-1, keepdim=True).clamp(min=1)
            out = self.node_mlp(torch.cat([node_feat, edge_feat_agg], dim=-1))
            if self.is_residual:
                out = node_feat + out
            out = out * self.n_nodes2mask(n_nodes).unsqueeze(-1)
            
        # * return out tensor
        return out
    
    # * converts coord to radial
    def coord2radial(
        self,
        coord: torch.Tensor,
        edge_index: torch.LongTensor
    ):
        # * choose between sparse and dense versions
        # * SPARSE VERSION:
        if coord.ndim == 2:
            coord_diff = coord[edge_index[0]] - coord[edge_index[1]]
            coord_radial = torch.sum(coord_diff ** 2, 1, keepdim=True)
            if self.normalize:
                coord_diff = coord_diff / (torch.sqrt(coord_radial).detach() + 1)

        # * DENSE VERSION:
        else:
            coord_diff = (coord.unsqueeze(2) - coord.unsqueeze(1)) * edge_index.unsqueeze(-1)
            coord_radial = (coord_diff ** 2).sum(-1)
            if self.normalize:
                coord_diff = coord_diff / (torch.sqrt(coord_radial).detach() + 1).unsqueeze(-1)

        # * return out tensor
        return coord_diff, coord_radial
    
    
    def forward(
        self,
        coord: torch.Tensor,
        node_feat: torch.Tensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.Tensor = None,
        edge_attr: torch.Tensor = None,
        vel: torch.Tensor = None,
        n_nodes: torch.LongTensor = None
    ):
        # if coord has 3 (2) dims then input is dense (sparse) and providing n_nodes is (not) mandatory
        assert coord.ndim == 2 or n_nodes is not None
        # if self.has_vel is True then velocity must be provided
        assert not self.has_vel or vel is not None

        coord_diff, coord_radial = self.coord2radial(coord, edge_index)
        edge_feat = self.edge_model(node_feat, edge_index, coord_radial, edge_weight, edge_attr)

        # * return velocity if model has velocity
        if self.has_vel:
            coord, vel = self.coord_model(coord, coord_diff, edge_feat, edge_index, node_feat, vel)
            node_feat = self.node_model(node_feat, edge_feat, edge_index, n_nodes)
            return coord, node_feat, vel
        else:
            coord = self.coord_model(coord, coord_diff, edge_feat, edge_index)
            node_feat = self.node_model(node_feat, edge_feat, edge_index, n_nodes)
            return coord, node_feat
        
        
class EGNN(torch.nn.Module):
    def __init__(
        self,
        layers: List[EGCL]
    ):
        super(EGNN, self).__init__()
        self.layers = torch.nn.Sequential(*layers)
        
    def forward(
        self,
        coord: torch.Tensor,
        node_feat: torch.Tensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.Tensor = None,
        edge_attr: torch.Tensor = None,
        vel: torch.Tensor = None,
        n_nodes: torch.LongTensor = None
    ):
        out = None
        for layer in self.layers:
            out = layer(coord, node_feat, edge_index, edge_weight, edge_attr, vel, n_nodes)
            if len(out) == 3:
                coord, node_feat, vel = out
            else:
                coord, node_feat = out
        assert isinstance(out, tuple)
        return out