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
            torch.nn.Linear(args.message_dim, args.message_dim),
            act_func
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
        
        # create attention-mlp: R(m) -> R(1)
        if args.has_attention:
            self.attention_mlp = torch.nn.Sequential(
                torch.nn.Linear(args.message_dim, 1),
                torch.nn.Sigmoid()
            )
        
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
        collect: bool = False
    ):
        assert coords.shape[0] == hidden.shape[0]
        n_nodes = coords.shape[0]
        
        collect_dict = dict()

        # calculate coordinate differences and L2-norms
        coords_dif = coords[edges[0]] - coords[edges[1]]
        coords_l2 = torch.linalg.norm(coords_dif, dim=1, ord=2).unsqueeze(0).to(self.args.device)
        coords_l2 = coords_l2.reshape([coords_l2.shape[1], coords_l2.shape[0]])
        
        if collect: collect_dict['coords_dif'] = coords_dif.detach().clone().cpu()
        if collect: collect_dict['coords_l2'] = coords_l2.detach().clone().cpu()
        
        # calulate hidden for all edge node pairs
        h_i = hidden[edges[0]]
        h_j = hidden[edges[1]]
        
        if collect: collect_dict['h_i'] = h_i.detach().clone().cpu()
        if collect: collect_dict['h_j'] = h_j.detach().clone().cpu()
        
        # run message mlp
        message_mlp_input = torch.cat([h_i, h_j, coords_l2], dim=1).to(self.args.device)
        message_mlp_input = message_mlp_input
        # print (f'(dev) message_mlp_input.shape: {message_mlp_input.shape}')
        m_ij = self.message_mlp(message_mlp_input)
        # print (f'(dev) m_ij.shape: {m_ij.shape}')
        
        # run attention mlp
        if self.args.has_attention:
            m_ij = self.attention_mlp(m_ij) * m_ij
        
        if collect: collect_dict['message_mlp_input'] = message_mlp_input.detach().clone().cpu()
        if collect: collect_dict['m_ij'] = m_ij.detach().clone().cpu()
        
        # run coordinate mlp
        coord_mlp_out = self.coord_mlp(m_ij) # <- (all-edges, coordinate-data)
        coord_trans = coords_dif * coord_mlp_out # <- (all-edges, coordinate-data)
        coord_trans_matrix = torch.zeros([n_nodes, n_nodes, 3]).to(self.args.device)
        coord_trans_matrix[edges[0], edges[1]] = coord_trans # <- [node_i, node_j, coordinate_data]
        
        if collect: collect_dict['coord_mlp_out'] = coord_mlp_out.detach().clone().cpu()
        if collect: collect_dict['coord_trans'] = coord_trans.detach().clone().cpu()
        if collect: collect_dict['coord_trans_matrix'] = coord_trans_matrix.detach().clone().cpu()
        
        # calculate each node i's number of neighbors
        node_i_nbors = torch.zeros([n_nodes]).long()
        for i in range(n_nodes):
            node_i_nbors[i] = torch.sum(edges[1] == i).item()
    
        assert torch.sum(node_i_nbors == 0).item() == 0
        node_i_nbors = node_i_nbors.unsqueeze(1)
        node_i_nbors = torch.cat([node_i_nbors]*3, dim=1).to(self.args.device)
        
        if collect: collect_dict['node_i_nbors'] = node_i_nbors.detach().clone().cpu()
        
        # compute output coordinates
        trans_sum_i = torch.sum(coord_trans_matrix, dim=0)
        trans_sum_i = trans_sum_i / node_i_nbors
        coords_out = coords + trans_sum_i
        
        if collect: collect_dict['trans_sum_i'] = trans_sum_i.detach().clone().cpu()
        if collect: collect_dict['coords_out'] = coords_out.detach().clone().cpu()
        
        # run hidden mlp
        m_ij_matrix = torch.zeros([n_nodes, n_nodes, self.args.message_dim]).to(self.args.device)
        m_ij_matrix[edges[0], edges[1]] = m_ij
        
        if collect: collect_dict['m_ij_matrix'] = m_ij_matrix.detach().clone().cpu()
        
        m_i = torch.sum(m_ij_matrix, dim=0)
        
        if collect: collect_dict['m_i'] = m_i.detach().clone().cpu()
        
        hidden_mlp_input = torch.cat([hidden, m_i], dim=1)
        
        if collect: collect_dict['hidden_mlp_input'] = hidden_mlp_input.detach().clone().cpu()
        
        hidden_mlp_out = self.hidden_mlp(hidden_mlp_input)
        
        if collect: collect_dict['hidden_mlp_out'] = hidden_mlp_out.detach().clone().cpu()
        
        hidden_out = hidden + hidden_mlp_out
        
        if collect: collect_dict['hidden_out'] = hidden_out.detach().clone().cpu()
        
        return coords_out, hidden_out, collect_dict
    
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
        collect: bool = False
    ):
        for layer in self.layers:
            coords, hidden, collection = layer(coords, hidden, edges, collect)
        return coords, hidden, collection
    
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