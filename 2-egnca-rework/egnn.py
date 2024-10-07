from argparse import Namespace
import torch

class egc(torch.nn.Module):
    
    def __init__(
        self,
        args: Namespace
    ):
        super(egc, self).__init__()
        self.args = args
        act_func = torch.nn.Tanh()
        
        # create message-mlp: R(2h + 1) -> R(m)
        self.message_mlp = torch.nn.Sequential(
            torch.nn.Linear((args.hidden_dim * 2) + 1, args.message_dim),
            act_func,
            torch.nn.Linear(args.message_dim, args.message_dim)
        ).to(args.device)
        
        # we set final coords layer's bias to false:
        # - model will always go though (0, 0, 0)
        last_coord_layer = torch.nn.Linear(args.message_dim, 1, bias=False)
        last_coord_layer.weight.data.zero_()
        
        # create coord-mlp: R(m) -> R(1)
        self.coord_mlp = torch.nn.Sequential(
            torch.nn.Linear(args.message_dim, args.message_dim),
            act_func,
            last_coord_layer,
            torch.nn.Tanh() if args.has_attention else torch.nn.Identity()
        ).to(args.device)
        
        # create hidden-mlp: R(h+m) -> R(h)
        self.hidden_mlp = torch.nn.Sequential(
            torch.nn.Linear(args.hidden_dim + args.message_dim, args.message_dim),
            act_func,
            torch.nn.Linear(args.message_dim, args.hidden_dim)
        ).to(args.device)
        
    def forward(
        self,
        coords: torch.Tensor,
        hidden: torch.Tensor,
        edges: torch.LongTensor,
    ):
        assert coords.shape[0] == hidden.shape[0]
        n_nodes = coords.shape[0]
        
        # print (f'(dev) coords.shape: {coords.shape}')
        # print (f'(dev) hidden.shape: {hidden.shape}')
        # print (f'(dev) edges.shape: {edges.shape}')
        
        # calculate coordinate differences and L2-norms
        coords_dif = coords[edges[0]] - coords[edges[1]]
        coords_l2 = torch.linalg.norm(coords_dif, dim=1, ord=2).unsqueeze(0).to(self.args.device)
        coords_l2 = coords_l2.reshape([coords_l2.shape[1], coords_l2.shape[0]])
        
        # calulate hidden for all edge node pairs
        h_i = hidden[edges[0]]
        h_j = hidden[edges[1]]
        
        # run message mlp
        message_mlp_input = torch.cat([coords_l2, h_i, h_j], dim=1).to(self.args.device)
        m_ij = self.message_mlp(message_mlp_input)
        
        # run coordinate mlp
        coord_trans = coords_dif * self.coord_mlp(m_ij) # <- (all-edges, coordinate-data)
        coord_trans_matrix = torch.zeros([n_nodes, n_nodes, 3]).to(self.args.device)
        coord_trans_matrix[edges[0], edges[1]] = coord_trans # <- [node_i, node_j, coordinate_data]
        
        # calculate each node i's number of neighbors
        node_i_nbors = torch.zeros([n_nodes]).long()
        for i in range(n_nodes):
            node_i_nbors[i] = torch.sum(edges[1] == i).item()
        assert torch.sum(node_i_nbors == 0).item() == 0
        node_i_nbors = node_i_nbors.unsqueeze(1)
        node_i_nbors = torch.cat([node_i_nbors]*3, dim=1).to(self.args.device)
        
        # compute output coordinates
        trans_sum_i = torch.sum(coord_trans_matrix, dim=0)
        trans_sum_i = trans_sum_i / node_i_nbors
        coords_out = coords + trans_sum_i
        
        # run hidden mlp
        m_ij_matrix = torch.zeros([n_nodes, n_nodes, self.args.message_dim]).to(self.args.device)
        print (f'(dev) m_ij_matrix.shape: {m_ij_matrix.shape}')
        print (f'(dev) m_ij.shape: {m_ij.shape}')
        m_ij_matrix[edges[0], edges[1]] = m_ij
        m_i = torch.sum(m_ij_matrix, dim=0)
        print (f'(dev) m_i.shape: {m_i.shape}')
        hidden_mlp_input = torch.cat([hidden, m_i], dim=1)
        print (f'(dev) hidden_mlp_input.shape: {hidden_mlp_input.shape}')
        h_i = self.hidden_mlp(hidden_mlp_input)
        print (f'(dev) h_i.shape: {h_i.shape}')
        print (f'(dev) hidden.shape: {hidden.shape}')
        hidden_out = hidden + h_i
        
        return coords_out, hidden_out
    
class egnn(torch.nn.Module):
    
    def __init__(
        self,
        layers: list[egc]
    ):
        super(egnn, self).__init__()
        self.layers = torch.nn.Sequential(*layers)
        
    def forward(
        self,
        coords: torch.Tensor,
        hidden: torch.Tensor,
        edges: torch.LongTensor,
    ):
        for layer in self.layers:
            coords, hidden = layer(coords, hidden, edges)
        return coords, hidden
    
def test_egnn_equivariance(
    n_tests: int = 100,
    verbose: bool = False
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from utils import default_namespace
    args = default_namespace()
    args.device = device
    my_egnn = egnn([
        egc(args),
        egc(args)
    ]).to(device)
    
    n_nodes = 16
    node_feat = torch.randn(n_nodes, n_nodes).to(device)
    
    W = torch.randn(n_nodes, n_nodes).sigmoid().to(device)
    W = (torch.tril(W) + torch.tril(W, -1).T)
    e_index = (W.fill_diagonal_(0) > 0.5).nonzero().T
    
    for i in range(n_tests):
        rotation = torch.nn.init.orthogonal_(torch.empty(3, 3)).to(device)
        translation = torch.randn(1, 3).to(device)

        in_coord_1 = torch.randn(n_nodes, 3).to(device)
        in_coord_2 = torch.matmul(rotation, in_coord_1.T).T + translation
        
        out_coord_1, _ = my_egnn(in_coord_1, node_feat, e_index)
        out_coord_2, _ = my_egnn(in_coord_2, node_feat, e_index)

        out_coord_1_aug = torch.matmul(rotation, out_coord_1.T).T + translation
        assert torch.allclose(out_coord_2, out_coord_1_aug, atol=1e-6)
        if (verbose): print(f'[egnn.py] test {i} complete')

    if (verbose): print(f'[egnn.py] finished running {n_tests} tests -- all succeeded')