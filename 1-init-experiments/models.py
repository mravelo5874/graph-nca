from torch_geometric.nn import PairNorm
from argparse import Namespace
from pool import TrainPool
from egnn import egnn, egc
import numpy as np
import torch

class MorphogenicEGNCA(torch.nn.Module):
    def __init__(
        self,
    ):
        super(MorphogenicEGNCA, self).__init__()
        
    def forward(
        self,
    ):
        pass
    
class FixedTargetEGNCA(torch.nn.Module):
    def __init__(
        self,
        args: Namespace
    ):
        super(FixedTargetEGNCA, self).__init__()
        self.args = args
        self.register_buffer('target_coords', args.target_coords * args.scale)
        self.register_buffer('target_edges', args.target_edges)
        
        # * [TODO] change this later
        seed_coords = args.target_coords
        seed_hidden = torch.rand([seed_coords.shape[0], args.hidden_dim])
        
        self.normalise = PairNorm(scale=1.0)
        self.mse = torch.nn.MSELoss(reduction='none')
        
        self.egnn = egnn([
            egc(args.coords_dim, 
                args.hidden_dim, 
                args.message_dim,
                has_attention=args.has_attention),
        ]).to(args.device)
        
        self.pool = TrainPool(
            pool_size=args.pool_size,
            seed_coords=seed_coords,
            seed_hidden=seed_hidden,
        )
        
        self.optimizer = torch.optim.Adam(
            self.egnn.parameters(), 
            lr=self.args.lr, 
            betas=(self.args.b1, self.args.b2), 
            weight_decay=self.args.wd
        )
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer,
            factor=self.args.factor_sch,
            patience=self.args.patience_sch,
            min_lr=1e-5,
        )
        
    def forward(
        self,
        batch_coords: torch.Tensor,
        batch_hidden: torch.Tensor,
        edges: torch.LongTensor
    ):
        # print (f'batch_coords.shape: {batch_coords.shape}')
        # print (f'batch_hidden.shape: {batch_hidden.shape}')
        # print (f'edges.shape: {edges.shape}')
        
        out_coords, out_hidden = self.egnn(
            coords=batch_coords,
            hidden=batch_hidden,
            edges=edges,
        )
        out_hidden = self.normalise(out_hidden)
        return out_coords, out_hidden
    
    def train(
        self,
        verbose: bool=False,
    ):
        epochs = self.args.epochs+1
        for i in range(epochs):
            batch_ids, \
            batch_coords, \
            batch_hidden, \
            rand_edges, \
            rand_edges_lens = self.pool.get_batch(self.args.batch_size)
            
            # * run for n steps
            n_steps = np.random.randint(self.args.min_steps, self.args.max_steps+1)
            for _ in range(n_steps):
                batch_coords, batch_hidden = self.forward(batch_coords, batch_hidden, self.target_edges.to(self.args.device))
            
            # * calculate loss
            edge_lens = torch.norm(batch_coords[rand_edges[0]] - batch_coords[rand_edges[1]], dim=-1)
        
            # print (f'edge_weight.dtype: {edge_weight.dtype}')
            # print (f'edge_weight.device: {edge_weight.device}')
            # print (f'rand_edge_weight.dtype: {rand_edge_weight.dtype}')
            # print (f'rand_edge_weight.device: {rand_edge_weight.device}')
            
            loss_per_edge = self.mse(edge_lens, rand_edges_lens)
            loss_per_graph = torch.stack([lpe.mean() for lpe in loss_per_edge.chunk(self.args.batch_size)])
            loss = loss_per_graph.mean()
        
            # * backward pass
            with torch.no_grad():
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step(loss)
                self.pool.update_pool(
                    self.args.batch_size, 
                    batch_ids, 
                    batch_coords, 
                    batch_hidden
                )
                i_loss = loss.item()
                
                if i % self.args.info_rate == 0 and i!= 0:
                    if (verbose): print (f'[{i}/{epochs}]\tloss: {i_loss}')
                    
                if i % self.args.save_rate == 0 and i != 0:
                    # self.save_model('_checkpoints', model, _NAME_+'_cp'+str(i))
                    pass
                    
def test_model_training(
    num_tests: int = 10,
    verbose: bool = False
):
    torch.set_default_dtype(torch.float32)
    from data.generate import generate_bunny_graph
    seed_coords, seed_edges = generate_bunny_graph()
    
    for i in range(num_tests):
        
        pool_size = np.random.randint(4, 512)
    
        args = Namespace()
        args.scale = 1.0
        args.target_coords = seed_coords
        args.target_edges = seed_edges
        args.coords_dim = 3
        args.hidden_dim = np.random.randint(8, 16)
        args.message_dim = np.random.randint(8, 16)
        args.has_attention = True
        args.device = 'cuda'
        args.pool_size = pool_size
        args.batch_size = np.random.randint(1, pool_size // 2)
        args.epochs = np.random.randint(1, 12)
        args.min_steps = np.random.randint(1, 12)
        args.max_steps = np.random.randint(13, 24)
        args.info_rate = 1e100
        args.save_rate = 1e100
        args.lr = 1e-3
        args.b1 = 0.9
        args.b2 = 0.999
        args.wd = 1e-5
        args.patience_sch = 500
        args.factor_sch = 0.5
        args.density_rand_edge = 1.0
        
        # * create and train model
        fixed_target_egnca = FixedTargetEGNCA(args)
        fixed_target_egnca.train(verbose=False)
        
        if (verbose): print(f'[models.py] test {i} complete')
            
    print(f'[models.py] finished running {num_tests} tests -- all succeeded')