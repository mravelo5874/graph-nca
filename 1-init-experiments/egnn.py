from typing import List, Optional
import torch

class egc(torch.nn.Module):
    
    def __init__(
        self,
        hidden_dim: int,
        message_dim: int,
        out_hidden_dim: Optional[int] = None,
        edge_attr_dim: Optional[int] = 0,
        act_name: Optional[str] = 'silu',
        aggr_coord: Optional[str] = 'mean',
        aggr_hidden:  Optional[str] = 'sum',
        is_residual: Optional[bool] = False,
        has_attention: Optional[bool] = False,
        device: Optional[str] = 'cuda',
    ):
        super(egc, self).__init__()
        assert aggr_coord == 'mean' or aggr_coord == 'sum'
        assert aggr_hidden == 'mean' or aggr_hidden == 'sum'
        self.aggr_coord = aggr_coord
        self.aggr_hidden = aggr_hidden
        self.out_node_dim = hidden_dim if out_hidden_dim is None else out_hidden_dim
        self.is_residual = is_residual
        self.has_attention = has_attention
        self.device = device
        act = {'tanh': torch.nn.Tanh(), 'lrelu': torch.nn.LeakyReLU(), 'silu': torch.nn.SiLU()}[act_name]
        
        # * create mlps
        self.message_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim + hidden_dim + edge_attr_dim + 1, message_dim),
            act,
            torch.nn.Linear(message_dim, message_dim),
            act
        ).to(device)
        last_coords_layer = torch.nn.Linear(message_dim, 1, bias=False)
        last_coords_layer.weight.data.zero_()
        self.coords_mlp = torch.nn.Sequential(
            torch.nn.Linear(message_dim, message_dim),
            act,
            last_coords_layer,
            torch.nn.Tanh() if has_attention else torch.nn.Identity()
        ).to(device)
        self.hidden_mlp = torch.nn.Sequential(
            torch.nn.Linear(message_dim + hidden_dim, message_dim),
            act,
            torch.nn.Linear(message_dim, self.out_node_dim)
        ).to(device)
        if has_attention:
            self.attention_mlp = torch.nn.Sequential(
                torch.nn.Linear(message_dim, 1),
                torch.nn.Sigmoid()
            ).to(device)
  
    def get_coord_n2(
        self,
        coords: torch.Tensor,
        edges: torch.LongTensor,
    ):  
        coords_diff = coords[edges[0]] - coords[edges[1]]
        coords_n2 = torch.linalg.vector_norm(coords_diff, dim=1)
        return coords_diff.to(self.device), coords_n2.to(self.device)
    
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
        edges: torch.LongTensor,
        edge_weight: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        verbose: Optional[bool] = False,
    ):
        # # * concat all tensors 
        if edge_attr is not None:
            assert edge_attr.size(1) == self.e_attr_dim
            edge_feat = torch.cat([coords_n2, hidden[edges[0]], hidden[edges[1]], edge_attr], dim=1).to(self.device)
        else:
            h0 = hidden[edges[0]]
            h1 = hidden[edges[1]]
            coords_n2 = coords_n2.unsqueeze(1)
            if verbose: print (f'h0.shape: {h0.shape}')
            if verbose: print (f'h1.shape: {h1.shape}')
            if verbose: print (f'coords_n2.shape: {coords_n2.shape}')
            edge_feat = torch.cat([coords_n2, h0, h1], dim=1).to(self.device)
        
        if verbose: print ('------------------------------')
        if verbose: print (f'(run_message_mlp) edge_feat.shape: {edge_feat.shape}, edge_feat:\n{edge_feat}')
        if verbose: print ('------------------------------')

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
        edges: torch.LongTensor,
        verbose: Optional[bool] = False,
    ):
        trans = coords_diff * self.coords_mlp(edge_feat)
        
        if verbose: print (f'trans.shape: {trans.shape}, trans:\n{trans}')
        
        num_nodes = coords.shape[0]
        trans_coord_adj_mat = torch.zeros([num_nodes, num_nodes, 3]).to(self.device)
        trans_coord_adj_mat[edges[0], edges[1]] = trans
        if verbose: print (f'trans_coord_adj_mat.shape: {trans_coord_adj_mat.shape}, trans_coord_adj_mat:\n{trans_coord_adj_mat}')
        
        # * calculate num neighbors per node
        n_neighbors = torch.zeros([num_nodes]).long()
        for i in range(num_nodes):
            n_neighbors[i] = torch.sum(edges[0] == i).item()
        assert torch.sum(n_neighbors == 0).item() == 0
        n_neighbors = n_neighbors.unsqueeze(1)
        n_neighbors = torch.cat([n_neighbors, n_neighbors, n_neighbors], dim=1).to(self.device)
        if verbose: print (f'n_neighbors.shape: {n_neighbors.shape}, n_neighbors:\n{n_neighbors}')
        
        sum_cols = torch.sum(trans_coord_adj_mat, dim=0).div(n_neighbors)
        sum_rows = torch.sum(trans_coord_adj_mat, dim=1).div(n_neighbors)
        if verbose: print (f'sum_cols.shape: {sum_cols.shape}, sum_cols:\n{sum_cols}')
        if verbose: print (f'sum_rows.shape: {sum_rows.shape}, sum_rows:\n{sum_rows}')
        # assert torch.equal(sum_cols.abs(), sum_rows.abs())
        
        # coord_agg = self.aggregated_sum(trans, edges[0], coords.size(0), mean=self.aggr_coord == 'mean')
        
        coords = coords + sum_cols
        return coords
    
    def run_hidden_mlp(
        self,
        hidden: torch.Tensor,
        edge_feat: torch.Tensor,
        edges: torch.LongTensor,
        verbose: Optional[bool] = False,
    ):
        num_nodes = hidden.shape[0]
        edge_feat_dim = edge_feat.shape[1]
        edge_feat_adj_mat = torch.zeros([num_nodes, num_nodes, edge_feat_dim]).to(self.device)
        edge_feat_adj_mat[edges[0], edges[1]] = edge_feat
        if verbose: print (f'edge_feat_adj_mat.shape: {edge_feat_adj_mat.shape}, edge_feat_adj_mat:\n{edge_feat_adj_mat}')
        
        sum_cols = torch.sum(edge_feat_adj_mat, dim=0)
        sum_rows = torch.sum(edge_feat_adj_mat, dim=1)
        if verbose: print (f'sum_cols.shape: {sum_cols.shape}, sum_cols:\n{sum_cols}')
        if verbose: print (f'sum_rows.shape: {sum_rows.shape}, sum_rows:\n{sum_rows}')
        # assert torch.equal(sum_cols.abs(), sum_rows.abs())
        
        # edge_feat_agg = self.aggregated_sum(edge_feat, edges[0], hidden.size(0), mean=self.aggr_hidden == 'mean')
        input_hidden_mlp = torch.cat([hidden, sum_cols], dim=1)
        if verbose: print (f'input_hidden_mlp.shape: {input_hidden_mlp.shape}, input_hidden_mlp:\n{input_hidden_mlp}')
        out = self.hidden_mlp(input_hidden_mlp)
        if self.is_residual:
            out = hidden + out
        return out

    def forward(
        self,
        coords: torch.Tensor,
        hidden: torch.Tensor,
        edges: torch.LongTensor,
        edge_weight: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        verbose: Optional[bool] = False
    ):
        if verbose: print ('------------------------------')
        if verbose:print (f'coords.shape: {coords.shape}, coords:\n{coords}')
        if verbose:print (f'hidden.shape: {hidden.shape}, hidden:\n{hidden}')
        if verbose: print (f'edges.shape: {edges.shape}, edges:\n{edges}')
        if verbose: print ('------------------------------')
        
        # * flatten out batches in coords / hidden tensors
        assert coords.shape[0] == hidden.shape[0]
        batch_size = coords.shape[0]
        num_nodes = coords.shape[1]
        coords_flat = coords.reshape([batch_size*coords.shape[1], coords.shape[2]])
        hidden_flat = hidden.reshape([batch_size*hidden.shape[1], hidden.shape[2]])
        
        # * expand edge tensors for all batches
        edges_flat = []
        for i in range(batch_size):
            batch_edges = edges.clone().add(i*num_nodes)
            edges_flat.append(batch_edges)
        edges_flat = torch.cat(edges_flat, dim=1).to(self.device)
        
        if verbose: print ('------------------------------')
        if verbose: print (f'coords_flat.shape: {coords_flat.shape}, coords_flat:\n{coords_flat}')
        if verbose: print (f'hidden_flat.shape: {hidden_flat.shape}, hidden_flat:\n{hidden_flat}')
        if verbose: print (f'edges_flat.shape: {edges_flat.shape}, edges_flat:\n{edges_flat}')
        if verbose: print ('------------------------------')
        
        coords_diff, coords_n2 = self.get_coord_n2(coords_flat, edges_flat)
        
        if verbose: print ('------------------------------')
        if verbose: print (f'coords_diff.shape: {coords_diff.shape}, coords_diff:\n{coords_diff}')
        if verbose: print (f'coords_n2.shape: {coords_n2.shape}, coords_n2:\n{coords_n2}')
        if verbose: print ('------------------------------')
        
        edge_feat = self.run_message_mlp(
            hidden=hidden_flat,
            coords_n2=coords_n2, 
            edges=edges_flat, 
            edge_weight=None, 
            edge_attr=None, 
            verbose=verbose
        )
        
        if verbose: print ('------------------------------')
        if verbose: print (f'edge_feat.shape: {edge_feat.shape}, edge_feat:\n{edge_feat}')
        if verbose: print ('------------------------------')
        
        coords_out = self.run_coords_mlp(
            coords=coords_flat, 
            coords_diff=coords_diff, 
            edge_feat=edge_feat, 
            edges=edges_flat,
            verbose=verbose
        )
        hidden_out = self.run_hidden_mlp(
            hidden=hidden_flat, 
            edge_feat=edge_feat,
            edges=edges_flat,
            verbose=verbose
        )
        
        if verbose: print ('------------------------------')
        if verbose: print (f'coords_out.shape: {coords_out.shape}, coords_out:\n{coords_out}')
        if verbose: print (f'hidden_out.shape: {hidden_out.shape}, hidden_out:\n{hidden_out}')
        if verbose: print ('------------------------------')
        
        # * reshape coords / hidden into batches
        coords_out = coords_out.reshape([batch_size, coords_out.shape[0]//batch_size, coords_out.shape[1]])
        hidden_out = hidden_out.reshape([batch_size, hidden_out.shape[0]//batch_size, hidden_out.shape[1]])
        
        if verbose: print ('------------------------------')
        if verbose: print (f'(returning) coords_out.shape: {coords_out.shape}, coords_out:\n{coords_out}')
        if verbose: print (f'(returning) hidden_out.shape: {hidden_out.shape}, hidden_out:\n{hidden_out}')
        if verbose: print ('------------------------------')
        
        return coords_out, hidden_out

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
        edges: torch.LongTensor,
        edge_weight: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        verbose: Optional[bool] = False,
    ):
        edges = edges.long()
        out = None
        for layer in self.layers:
            out = layer(coords, hidden, edges, edge_weight, edge_attr, verbose)
            coords, hidden = out
        assert isinstance(out, tuple)
        return out
    
def test_egnn_equivariance(
    num_tests: int = 100,
    verbose: bool = False
):

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
    
    for i in range(num_tests):

        rotation = torch.nn.init.orthogonal_(torch.empty(coords_dim, coords_dim)).to(device)
        translation = torch.randn(1, coords_dim).to(device)

        in_coord_1 = torch.randn(num_nodes, coords_dim).to(device)
        in_coord_2 = torch.matmul(rotation, in_coord_1.T).T + translation
        
        out_coord_1, _ = my_egnn(in_coord_1, node_feat, e_index)
        out_coord_2, _ = my_egnn(in_coord_2, node_feat, e_index)

        out_coord_1_aug = torch.matmul(rotation, out_coord_1.T).T + translation
        assert torch.allclose(out_coord_2, out_coord_1_aug, atol=1e-6)
        
        if (verbose): print(f'[pool.py] test {i} complete')

    if (verbose): print(f'[egnn.py] finished running {num_tests} tests -- all succeeded')