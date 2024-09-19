import torch
from typing import List, Optional

class egc(torch.nn.Module):
    
    def __init__(
        self,
        x_coord_dim: int,
        hidden_dim: int,
        message_dim: int,
        out_hidden_dim: Optional[int] = None,
        e_attr_dim: Optional[int] = 0,
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
            torch.nn.Linear(hidden_dim + hidden_dim + e_attr_dim + 1, message_dim),
            act,
            torch.nn.Linear(message_dim, message_dim),
            act
        )
        last_coord_layer = torch.nn.Linear(x_coord_dim, 1, bias=False)
        last_coord_layer.weight.data.zero_()
        self.x_coords_mlp = torch.nn.Sequential(
            torch.nn.Linear(message_dim, x_coord_dim),
            act,
            last_coord_layer
        )
        self.x_hidden_mlp = torch.nn.Sequential(
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
        x_coords: torch.Tensor,
        e_index: torch.LongTensor,
    ):
        x_coord_diff = x_coords[e_index[0]] - x_coords[e_index[1]]
        x_coord_n2 = torch.sum(x_coord_diff ** 2, 1, keepdim=True)
        if self.norm_n2:
            x_coord_diff = x_coord_diff / (torch.sqrt(x_coord_n2).detach() + 1)
        return x_coord_diff, x_coord_n2
    
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
        x_hidden: torch.Tensor,
        x_coord_n2: torch.Tensor,
        e_index: torch.LongTensor,
        e_weight: Optional[torch.Tensor] = None,
        e_attr: Optional[torch.Tensor] = None,
    ):
        # # * concat all tensors 
        if e_attr is not None:
            assert e_attr.size(1) == self.e_attr_dim
            edge_feat = torch.cat([x_hidden[e_index[0]], x_hidden[e_index[1]], x_coord_n2, e_attr], dim=1)
        else:
            edge_feat = torch.cat([x_hidden[e_index[0]], x_hidden[e_index[1]], x_coord_n2], dim=1)

        # * run mlp
        out = self.message_mlp(edge_feat)
        
        # * apply edge weights
        if e_weight is not None:
            out = e_weight.unsqueeze(1) * out
        if self.has_attention:
            out = self.attention_mlp(out) * out

        return out
    
    def run_x_coords_mlp(
        self,
        x_coords: torch.Tensor,
        x_coords_diff: torch.Tensor,
        e_feat: torch.Tensor,
        e_index: torch.LongTensor,
    ):
        trans = x_coords_diff * self.x_coords_mlp(e_feat)
        coord_agg = self.aggregated_sum(trans, e_index[0], x_coords.size(0), mean=self.aggr_coord == 'mean')
        x_coords = x_coords + coord_agg
        return x_coords
    
    def run_x_hidden_mlp(
        self,
        x_hidden: torch.Tensor,
        e_feat: torch.Tensor,
        e_index: torch.LongTensor,
    ):
        edge_feat_agg = self.aggregated_sum(e_feat, e_index[0], x_hidden.size(0), mean=self.aggr_hidden == 'mean')
        out = self.x_hidden_mlp(torch.cat([x_hidden, edge_feat_agg], dim=1))
        if self.is_residual:
            out = x_hidden + out
        return out

    def forward(
        self,
        x_coords: torch.Tensor,
        x_hidden: torch.Tensor,
        e_index: torch.LongTensor,
        e_weight: Optional[torch.Tensor] = None,
        e_attr: Optional[torch.Tensor] = None
    ):
        x_coords_diff, x_coord_n2 = self.get_coord_n2(x_coords, e_index)
        e_feat = self.run_message_mlp(x_hidden, x_coord_n2, e_index, e_weight, e_attr)
        x_coords = self.run_x_coords_mlp(x_coords, x_coords_diff, e_feat, e_index)
        x_hidden = self.run_x_hidden_mlp(x_hidden, e_feat, e_index)
        return x_coords, x_hidden

class egnn(torch.nn.Module):

    def __init__(
        self,
        layers: List[egc]
    ):
        super(egnn, self).__init__()
        self.layers = torch.nn.Sequential(*layers)

    def forward(
        self,
        x_coords: torch.Tensor,
        x_hidden: torch.Tensor,
        e_index: torch.LongTensor,
        e_weight: Optional[torch.Tensor] = None,
        e_attr: Optional[torch.Tensor] = None,
    ):
        out = None
        for layer in self.layers:
            out = layer(x_coords, x_hidden, e_index, e_weight, e_attr)
            x_coords, x_hidden = out
        assert isinstance(out, tuple)
        return out
    
def test_egnn_equivariance():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_nodes, hidden_dim, message_dim, x_coord_dim = 6, 8, 16, 3
    my_egnn = egnn([
        egc(x_coord_dim, hidden_dim, message_dim, has_attention=True),
        egc(x_coord_dim, hidden_dim, message_dim, has_attention=True)
    ]).to(device)
    node_feat = torch.randn(n_nodes, hidden_dim).to(device)
    
    W = torch.randn(n_nodes, n_nodes).sigmoid().to(device)
    W = (torch.tril(W) + torch.tril(W, -1).T)
    e_index = (W.fill_diagonal_(0) > 0.5).nonzero().T
    
    num_tests = 100
    for i in range(num_tests):

        rotation = torch.nn.init.orthogonal_(torch.empty(x_coord_dim, x_coord_dim)).to(device)
        translation = torch.randn(1, x_coord_dim).to(device)

        in_coord_1 = torch.randn(n_nodes, x_coord_dim).to(device)
        in_coord_2 = torch.matmul(rotation, in_coord_1.T).T + translation
        
        out_coord_1, _ = my_egnn(in_coord_1, node_feat, e_index)
        out_coord_2, _ = my_egnn(in_coord_2, node_feat, e_index)

        out_coord_1_aug = torch.matmul(rotation, out_coord_1.T).T + translation
        assert torch.allclose(out_coord_2, out_coord_1_aug, atol=1e-6)

    print(f'[egnn.py] finished running {num_tests} tests -- all succeeded')