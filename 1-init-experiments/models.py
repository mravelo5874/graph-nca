from data.generate import generate_line_graph
from torch_geometric.nn import PairNorm
from argparse import Namespace
from pool import TrainPool
from egnn import egnn, egc
import numpy as np
import datetime
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
        
        if args.graph == 'line':
            target_coords, target_edges = generate_line_graph(8)
        
        self.register_buffer('target_coords', target_coords * args.scale)
        self.register_buffer('target_edges', target_edges)
        
        # * setup seed coords/hidden
        num_nodes = target_coords.size(0)
        seed_coords = torch.empty(num_nodes, args.coords_dim).normal_(std=0.5)
        seed_hidden = torch.rand([seed_coords.shape[0], args.hidden_dim])
        
        # * [TODO] save seed data to file !!!
        
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
        out_coords, out_hidden = self.egnn(
            coords=batch_coords,
            hidden=batch_hidden,
            edges=edges,
        )
        out_hidden = self.normalise(out_hidden)
        return out_coords, out_hidden
    
    def save(
        self,
        path: str,
        name: str
    ):
        import os
        import json
        
        # * create directory
        if not os.path.exists(path):
            os.makedirs(path)
            
        # * save model
        torch.save(self.state_dict(), f'{path}/{name}.pt')
        
        # * save args
        with open(f'{path}/args.txt', 'w') as f:
            json.dump(self.args.__dict__, f, indent=2, default=lambda o: '<not serializable>')
            
        print (f'[models.py] saved model to: {path}')
        
    def run_for(
        self,
        num_steps: int,
    ):
        coords, hidden = self.pool.get_seed()
        coords = coords.to(self.args.device)
        hidden = hidden.to(self.args.device)
        edges = self.target_edges.clone().to(self.args.device)
        
        with torch.no_grad():
            for _ in range(num_steps):
                coords, hidden = self.forward(coords, hidden, edges)
                
            # * calculate loss
            edge_lens = torch.norm(coords[edges[0]] - coords[edges[1]], dim=-1)
            loss_per_edge = self.mse(edge_lens, edges)
            loss_per_graph = torch.stack([lpe.mean() for lpe in loss_per_edge])
            loss = loss_per_graph.mean()
            
        return coords, edges, loss.item()
    
    def train(
        self,
        verbose: bool=False,
    ):
        train_start = datetime.datetime.now()
        self.loss_log = []
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
                
                loss = loss.item() * 1_000_000
                
                if i!= 0.0:
                    self.loss_log.append(loss)
                
                if i % self.args.info_rate == 0 and i!= 0:
                    secs = (datetime.datetime.now()-train_start).seconds
                    if secs == 0: secs = 1
                    time = str(datetime.timedelta(seconds=secs))
                    iter_per_sec = float(i)/float(secs)
                    est_time_sec = int((epochs-i)*(1/iter_per_sec))
                    est = str(datetime.timedelta(seconds=est_time_sec))
                    avg = sum(self.loss_log[-self.args.info_rate:])/float(self.args.info_rate)
                    lr = np.round(self.lr_scheduler.get_last_lr()[0], 8)
                    
                    if (verbose): 
                        print (f'[{i}/{epochs}]\t {np.round(iter_per_sec, 3)}it/s\t time: {time}~{est}\t loss: {np.round(avg, 6)}>{np.round(np.min(self.loss_log), 6)}\t lr: {lr}')
                    
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