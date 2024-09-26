from collections.abc import Callable
from typing import Optional
import numpy as np
import torch

class TrainPool:
    
    def __init__(
        self,
        pool_size: int,
        hidden_dim: int,
        seed_coords: torch.Tensor,
        target_coords: torch.Tensor,
        loss_func: Callable[[torch.Tensor, torch.Tensor], float],
        rand_edge_percent: float = 1.0,
        device: Optional[str] = 'cuda',
        reset_at: Optional[int] = float('inf')
    ):
        self.pool_size = pool_size
        self.loss_func = loss_func
        self.device = device
        self.reset_at = reset_at
        seed_hidden = torch.ones([seed_coords.shape[0], hidden_dim])
        self.seed = (seed_coords.clone(), seed_hidden.clone())
        self.reset()
        
        # * dataset stuff
        self.rand_edge_percent = rand_edge_percent
        self.num_nodes = seed_coords.size(0)
        self.all_edges = torch.ones(self.num_nodes, self.num_nodes).tril(-1).nonzero().T
        row, col = self.all_edges[0], self.all_edges[1]
        self.all_edge_lens = torch.norm(target_coords[row] - target_coords[col], dim=-1)
        self.num_rand_edges = int(self.rand_edge_percent * self.all_edges.size(1))
        
    def reset(self):
        seed_coords, seed_hidden = self.seed
        seed_coords = seed_coords.clone().to('cpu')
        seed_hidden = seed_hidden.clone().to('cpu')
        self.cache = dict()
        self.cache['coords'] = seed_coords.clone().repeat([self.pool_size, 1, 1])
        self.cache['hidden'] = seed_hidden.clone().repeat([self.pool_size, 1, 1])
        self.cache['steps'] = [0] * self.pool_size
        self.cache['loss'] = torch.full((self.pool_size, ), torch.inf)
        
    def get_random_edges(self):
        perm = torch.randperm(self.all_edges.size(1))[:self.num_rand_edges]
        rand_target_edges = self.all_edges[:, perm].clone().to(self.device)
        rand_target_edges_lens = self.all_edge_lens[perm].clone().to(self.device)
        return rand_target_edges, rand_target_edges_lens
    
    def get_batch(
        self,
        batch_size: int,
        replace_lowest_loss: bool = True
    ):
        # * extract batch from pool
        batch_ids = np.random.choice(self.pool_size, batch_size, replace=False)
        batch_coords = self.cache['coords'][batch_ids].clone().to(self.device)
        batch_hidden = self.cache['hidden'][batch_ids].clone().to(self.device)
        
        # * get random edges
        rand_target_edges, rand_target_edges_lens = self.get_random_edges()
        rand_edges = []
        rand_edges_lens = []
        for i in range(batch_size):
            batch_edges = rand_target_edges.clone().add(i*self.num_nodes)
            rand_edges.append(batch_edges)
            rand_edges_lens.append(rand_target_edges_lens)
        rand_edges = torch.cat(rand_edges, dim=1).to(self.device)
        rand_edges_lens = torch.cat(rand_edges_lens, dim=0).to(self.device)
        
        # * re-add seed into batch (highest loss)
        if replace_lowest_loss:
            max_loss_id = self.cache['loss'][batch_ids].argmax().item()
            seed_coords, seed_hidden = self.seed
            batch_coords[max_loss_id] = seed_coords.clone().to(self.device)
            batch_hidden[max_loss_id] = seed_hidden.clone().to(self.device)
            self.cache['steps'][max_loss_id] = 0
            
        # * check for reset graphs (re-add seed)
        for i, id in enumerate(batch_ids):
            if self.cache['steps'][id] > self.reset_at:
                batch_coords[i] = seed_coords.clone().to(self.device)
                batch_hidden[i] = seed_hidden.clone().to(self.device)
                self.cache['steps'][id] = 0
                    
        # * squish batches into one dim
        # batch_coords_rs = batch_coords.reshape([batch_size*batch_coords.shape[1], batch_coords.shape[2]])
        # batch_hidden_rs = batch_hidden.reshape([batch_size*batch_hidden.shape[1], batch_hidden.shape[2]])
        
        batch_coords = batch_coords.view(-1, batch_coords.shape[2])
        batch_hidden = batch_hidden.view(-1, batch_hidden.shape[2])
        
        # assert torch.equal(batch_coords_rs, batch_coords)
        # assert torch.equal(batch_hidden_rs, batch_hidden)
        
        return batch_ids, batch_coords, batch_hidden, rand_edges, rand_edges_lens
    
    def update_pool(
        self,
        batch_size: int,
        batch_ids: np.ndarray[int],
        batch_coords: torch.Tensor,
        batch_hidden: torch.Tensor,
        batch_loss: torch.Tensor,
        steps: int,
    ):
        # * unsquish batches
        batch_coords = batch_coords.reshape([batch_size, batch_coords.shape[0]//batch_size, batch_coords.shape[1]])
        batch_hidden = batch_hidden.reshape([batch_size, batch_hidden.shape[0]//batch_size, batch_hidden.shape[1]])
        
        # * replace in pool cache
        self.cache['coords'][batch_ids] = batch_coords.detach().cpu()
        self.cache['hidden'][batch_ids] = batch_hidden.detach().cpu()
        self.cache['loss'][batch_ids] = batch_loss.detach().cpu()
        
        # * update steps
        for i in batch_ids:
            self.cache['steps'][i] += steps
        
def test_pool_functionality(
    num_tests: int = 100,
    verbose: bool = False
):
    from data.generate import generate_bunny_graph
    
    seed_coords, _ = generate_bunny_graph()
    seed_hidden = torch.rand([seed_coords.shape[0], 16])
    
    for i in range(num_tests):
        
        pool = TrainPool(
            pool_size=np.random.randint(4, 512),
            seed_coords=seed_coords,
            seed_hidden=seed_hidden
        )
        
        batch_size = np.random.randint(1, pool.pool_size // 2)
        batch_ids, \
        batch_coords, \
        batch_hidden, \
        rand_edges, \
        rand_edges_lens = pool.get_batch(batch_size)
        batch_coords = batch_coords + torch.rand_like(batch_coords)
        batch_hidden = batch_hidden + torch.rand_like(batch_hidden)
        pool.update_pool(batch_size, batch_ids, batch_coords, batch_hidden)
        
        clean_ids = []
        for id in batch_ids:
            assert not torch.equal(pool.cache['coords'][id], seed_coords)
            assert not torch.equal(pool.cache['hidden'][id], seed_hidden)
            
            # * generate list of clean ids
            for _ in range(pool.pool_size):
                id = (id + 1) % pool.pool_size
                if id not in batch_ids and id not in clean_ids:
                    break
            clean_ids.append(id)
            
        for id in clean_ids:
            assert torch.equal(pool.cache['coords'][id], seed_coords)
            assert torch.equal(pool.cache['hidden'][id], seed_hidden)
            
        if (verbose): print(f'[pool.py] test {i} complete')
            
    print(f'[pool.py] finished running {num_tests} tests -- all succeeded')