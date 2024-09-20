import torch
from typing import List, Optional

class egc(torch.nn.Module):
    
    def __init__(
        self,
        coords_dim: int,
        hidden_dim: int,
        message_dim: int,
        out_hidden_dim: Optional[int] = None,
        edge_attr_dim: Optional[int] = 0,
        act_name: Optional[str] = 'silu',
        aggr_coord: Optional[str] = 'mean',
        aggr_hidden:  Optional[str] = 'sum',
        norm_n2: Optional[bool] = False,
        is_residual: Optional[bool] = False,
        has_attention: Optional[bool] = False,
    ):
        super(egc, self).__init__()
        assert aggr_coord == 'mean' or aggr_coord == 'sum'
        assert aggr_hidden == 'mean' or aggr_hidden == 'sum'
        self.aggr_coord = aggr_coord
        self.aggr_hidden = aggr_hidden
        self.out_node_dim = hidden_dim if out_hidden_dim is None else out_hidden_dim
        self.norm_n2 = norm_n2
        self.is_residual = is_residual
        self.has_attention = has_attention
        act = {'tanh': torch.nn.Tanh(), 'lrelu': torch.nn.LeakyReLU(), 'silu': torch.nn.SiLU()}[act_name]
        
        # * create mlps
        self.message_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim + hidden_dim + edge_attr_dim + 1, message_dim),
            act,
            torch.nn.Linear(message_dim, message_dim),
            act
        )
        last_coords_layer = torch.nn.Linear(coords_dim, 1, bias=False)
        last_coords_layer.weight.data.zero_()
        self.coords_mlp = torch.nn.Sequential(
            torch.nn.Linear(message_dim, coords_dim),
            act,
            last_coords_layer
        )
        self.hidden_mlp = torch.nn.Sequential(
            torch.nn.Linear(message_dim + hidden_dim, message_dim),
            act,
            torch.nn.Linear(message_dim, self.out_node_dim)
        )
        if has_attention:
            self.attention_mlp = torch.nn.Sequential(
                torch.nn.Linear(message_dim, 1),
                torch.nn.Sigmoid()
            )
  
    def get_coord_n2(
        self,
        coords: torch.Tensor,
        e_index: torch.LongTensor,
    ):
        coords_diff = coords[e_index[0]] - coords[e_index[1]]
        coords_n2 = torch.sum(coords_diff ** 2, 1, keepdim=True)
        if self.norm_n2:
            coords_diff = coords_diff / (torch.sqrt(coords_n2).detach() + 1)
        return coords_diff, coords_n2
    
    def aggregated_sum(
        self,
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
        
    def run_message_mlp(
        self,
        hidden: torch.Tensor,
        coords_n2: torch.Tensor,
        edge_index: torch.LongTensor,
        edge_weight: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
    ):
        # # * concat all tensors 
        if edge_attr is not None:
            assert edge_attr.size(1) == self.e_attr_dim
            edge_feat = torch.cat([hidden[edge_index[0]], hidden[edge_index[1]], coords_n2, edge_attr], dim=1)
        else:
            edge_feat = torch.cat([hidden[edge_index[0]], hidden[edge_index[1]], coords_n2], dim=1)

        # * run mlp
        out = self.message_mlp(edge_feat)
        
        # * apply edge weights
        if edge_weight is not None:
            out = edge_weight.unsqueeze(1) * out
        if self.has_attention:
            out = self.attention_mlp(out) * out

        return out
    
    def run_coords_mlp(
        self,
        coords: torch.Tensor,
        coords_diff: torch.Tensor,
        edge_feat: torch.Tensor,
        edge_index: torch.LongTensor,
    ):
        trans = coords_diff * self.coords_mlp(edge_feat)
        coord_agg = self.aggregated_sum(trans, edge_index[0], coords.size(0), mean=self.aggr_coord == 'mean')
        coords = coords + coord_agg
        return coords
    
    def run_hidden_mlp(
        self,
        hidden: torch.Tensor,
        edge_feat: torch.Tensor,
        edge_index: torch.LongTensor,
    ):
        edge_feat_agg = self.aggregated_sum(edge_feat, edge_index[0], hidden.size(0), mean=self.aggr_hidden == 'mean')
        out = self.hidden_mlp(torch.cat([hidden, edge_feat_agg], dim=1))
        if self.is_residual:
            out = hidden + out
        return out

    def forward(
        self,
        coords: torch.Tensor,
        hidden: torch.Tensor,
        edge_index: torch.LongTensor,
        edge_weight: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None
    ):
        coords_diff, coords_n2 = self.get_coord_n2(coords, edge_index)
        edge_feat = self.run_message_mlp(hidden, coords_n2, edge_index, edge_weight, edge_attr)
        coords = self.run_coords_mlp(coords, coords_diff, edge_feat, edge_index)
        hidden = self.run_hidden_mlp(hidden, edge_feat, edge_index)
        return coords, hidden

class egnn(torch.nn.Module):

    def __init__(
        self,
        layers: List[egc]
    ):
        super(egnn, self).__init__()
        self.layers = torch.nn.Sequential(*layers)

    def forward(
        self,
        coords: torch.Tensor,
        hidden: torch.Tensor,
        edge_index: torch.LongTensor,
        edge_weight: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
    ):
        out = None
        for layer in self.layers:
            out = layer(coords, hidden, edge_index, edge_weight, edge_attr)
            coords, hidden = out
        assert isinstance(out, tuple)
        return out
    
def test_egnn_equivariance():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_nodes, hidden_dim, message_dim, coords_dim = 6, 8, 16, 3
    my_egnn = egnn([
        egc(coords_dim, hidden_dim, message_dim, has_attention=True),
        egc(coords_dim, hidden_dim, message_dim, has_attention=True)
    ]).to(device)
    node_feat = torch.randn(num_nodes, hidden_dim).to(device)
    
    W = torch.randn(num_nodes, num_nodes).sigmoid().to(device)
    W = (torch.tril(W) + torch.tril(W, -1).T)
    e_index = (W.fill_diagonal_(0) > 0.5).nonzero().T
    
    num_tests = 100
    for _ in range(num_tests):

        rotation = torch.nn.init.orthogonal_(torch.empty(coords_dim, coords_dim)).to(device)
        translation = torch.randn(1, coords_dim).to(device)

        in_coord_1 = torch.randn(num_nodes, coords_dim).to(device)
        in_coord_2 = torch.matmul(rotation, in_coord_1.T).T + translation
        
        out_coord_1, _ = my_egnn(in_coord_1, node_feat, e_index)
        out_coord_2, _ = my_egnn(in_coord_2, node_feat, e_index)

        out_coord_1_aug = torch.matmul(rotation, out_coord_1.T).T + translation
        assert torch.allclose(out_coord_2, out_coord_1_aug, atol=1e-6)

    print(f'[egnn.py] finished running {num_tests} tests -- all succeeded')