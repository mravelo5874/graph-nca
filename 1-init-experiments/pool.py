from typing import Optional
import numpy as np
import torch

class TrainPool:
    
    def __init__(
        self,
        pool_size: int,
        seed_coords: torch.Tensor,
        seed_hidden: torch.Tensor,
        rand_edge_percent: float = 1.0,
        device: Optional[str] = 'cuda'
    ):
        self.pool_size = pool_size
        self.device = device
        self.seed = (seed_coords.clone(), seed_hidden.clone())
        self.cache = dict()
        self.cache['coords'] = seed_coords.clone().repeat([pool_size, 1, 1])
        self.cache['hidden'] = seed_hidden.clone().repeat([pool_size, 1, 1])
        self.cache['iters'] = [0] * pool_size
        
        # * dataset stuff
        self.rand_edge_percent = rand_edge_percent
        self.num_nodes = seed_coords.size(0)
        self.all_edges = torch.ones(self.num_nodes, self.num_nodes).tril(-1).nonzero().T
        row, col = self.all_edges[0], self.all_edges[1]
        self.all_edge_lens = torch.norm(seed_coords[row] - seed_coords[col], dim=-1)
        self.num_rand_edges = int(self.rand_edge_percent * self.all_edges.size(1))
        
    def get_seed(
        self,
    ):
        return self.seed[0].clone(), self.seed[1].clone()
    
    def get_batch(
        self,
        batch_size: int,
        replace_lowest_loss: bool = True
    ):
        batch_ids = np.random.choice(self.pool_size, batch_size, replace=False)
        batch_coords = self.cache['coords'][batch_ids].clone().to(self.device)
        batch_hidden = self.cache['hidden'][batch_ids].clone().to(self.device)
        
        # * re-order batch based on loss
        if replace_lowest_loss:
            pass
            # loss_ranks = torch.argsort(voxutil.voxel_wise_loss_function(x, target_ten, _dims=[-1, -2, -3, -4]), descending=True)
            # x = x[loss_ranks]
            # # * re-add seed into batch
            # x[:1] = seed_ten
        
        # * squish batches into one dim
        batch_coords = batch_coords.reshape([batch_size*batch_coords.shape[1], batch_coords.shape[2]])
        batch_hidden = batch_hidden.reshape([batch_size*batch_hidden.shape[1], batch_hidden.shape[2]])

        # * get random edges
        perm = torch.randperm(self.all_edges.size(1))[:self.num_rand_edges]
        rand_edges = self.all_edges[:, perm]
        rand_edges.to(self.device)
        rand_edges_lens = self.all_edge_lens[perm]
        rand_edges_lens = rand_edges_lens.to(self.device)
            
        return batch_ids, batch_coords, batch_hidden, rand_edges, rand_edges_lens
    
    def update_pool(
        self,
        batch_size: int,
        batch_ids: np.ndarray[int],
        batch_coords: torch.Tensor,
        batch_hidden: torch.Tensor
    ):
        # * unsquish batches
        batch_coords = batch_coords.reshape([batch_size, batch_coords.shape[0]//batch_size, batch_coords.shape[1]])
        batch_hidden = batch_hidden.reshape([batch_size, batch_hidden.shape[0]//batch_size, batch_hidden.shape[1]])
        
        # * replace in pool cache
        self.cache['coords'][batch_ids] = batch_coords.detach().cpu()
        self.cache['hidden'][batch_ids] = batch_hidden.detach().cpu()
        
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