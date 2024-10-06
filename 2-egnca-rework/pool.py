from argparse import Namespace
from graph import graph
import numpy as np
import torch

class train_pool:
    
    def __init__(
        self,
        args: Namespace,
        seed_graph: graph,
        target_graph: graph
    ):
        self.args = args
        self.seed_graph = seed_graph
        seed_coords, _, seed_hidden = seed_graph.get()
        self.n_nodes = seed_coords.shape[0]
        
        # create cache
        self.cache = dict()
        self.cache['coords'] = seed_coords.repeat([args.pool_size, 1, 1])
        self.cache['hidden'] = seed_hidden.repeat([args.pool_size, 1, 1])
        self.cache['loss'] = np.array([np.inf]*args.pool_size, dtype=np.float16)
        self.cache['steps'] =  np.array([0]*args.pool_size, dtype=np.int16)
        
        # generate all-edges (V x V) matrix
        self.all_edges = torch.ones(self.n_nodes, self.n_nodes).tril(-1).nonzero().T
        row, col = self.all_edges[0], self.all_edges[1]
        target_coords, _, _ = target_graph.get()
        self.all_edges_lens = torch.norm(target_coords[row] - target_coords[col], dim=-1)
    
    def get_random_graph(self):
        id = np.random.randint(self.args.pool_size)
        data = dict()
        data['id'] = id
        coords = self.cache['coords'][id].clone().detach().cpu()
        hidden = self.cache['hidden'][id].clone().detach().cpu()
        data['coords'] = coords.squeeze(0)
        data['hidden'] = hidden.squeeze(0)
        data['steps'] = self.cache['steps'][id]
        data['loss'] = self.cache['loss'][id]
        return data

    def get_comp_edges(self, percent: float):
        n_comp_edges = int(percent * self.all_edges.shape[1])
        perm = torch.randperm(self.all_edges.size(1))[:n_comp_edges]
        comp_edges = self.all_edges[:, perm].detach().clone().to(self.args.device)
        comp_lens = self.all_edges_lens[perm].detach().clone().to(self.args.device)
        return comp_edges, comp_lens
        
    def get_batch(
        self, 
        batch_size: int,
        comp_edge_percent: float = 0.5,
        replace_max_loss: bool = True,
        apply_damage: bool = True,
    ) -> dict:
        
        # * extract batch data from cache
        batch_ids = np.random.choice(self.args.pool_size, batch_size, replace=False)
        batch_coords = self.cache['coords'][batch_ids].clone().detach().to(self.args.device)
        batch_hidden = self.cache['hidden'][batch_ids].clone().detach().to(self.args.device)
        comp_edges, comp_lens = self.get_comp_edges(comp_edge_percent)
        
        # replace highest loss graph with seed graph
        if replace_max_loss:
            seed_coords, _, seed_hidden = self.seed_graph.get()
            max_loss_id = self.cache['loss'][batch_ids].argmax().item()
            batch_coords[max_loss_id] = seed_coords.to(self.args.device)
            batch_hidden[max_loss_id] = seed_hidden.to(self.args.device)
            self.cache['steps'][max_loss_id] = 0
            self.cache['loss'][max_loss_id] = np.inf
            
        # apply damage
        if apply_damage:
            pass
        
        # flatten out coordinate / hidden batches
        batch_coords = batch_coords.reshape(batch_size * batch_coords.shape[1], 3)
        batch_hidden = batch_hidden.reshape(batch_size * batch_hidden.shape[1], batch_hidden.shape[2])
        
        # return batch-data object
        data = dict()
        data['ids'] = batch_ids
        data['coords'] = batch_coords
        data['hidden'] = batch_hidden
        data['comp_edges'] = comp_edges
        data['comp_lens'] = comp_lens
        return data

    def update(
        self,
        data: dict,
        steps: int,
        losses: np.ndarray[float]
    ):
        # reshape coordinate / hidden batches
        batch_size = len(data['ids'])
        batch_coords = data['coords'].clone().detach().cpu()
        batch_hidden = data['hidden'].clone().detach().cpu()
        batch_coords = batch_coords.reshape(batch_size, batch_coords.shape[0]//batch_size, 3)
        batch_hidden = batch_hidden.reshape(batch_size, batch_hidden.shape[0]//batch_size, batch_hidden.shape[1])
        
        # update cache values
        ids = data['ids']
        self.cache['coords'][ids] = batch_coords
        self.cache['hidden'][ids] = batch_hidden
        self.cache['steps'][ids] += np.array([steps]*len(ids), dtype=np.int16)
        self.cache['loss'][ids] = np.array(losses, dtype=np.float16)
        
        
def test_pool_functionality(
    num_tests: int = 100,
    verbose: bool = False
):
    from data.generate import retrieve_bunny_graph
    from utils import default_namespace
    
    args = default_namespace()
    target_coords, target_edges = retrieve_bunny_graph()
    seed_coords = torch.empty([target_coords.shape[0], 3]).normal_(std=args.seed_std)
    seed_graph = graph(seed_coords, target_edges)
    target_graph = graph(target_coords, target_edges)
    
    for i in range(num_tests):
        
        args.pool_size = np.random.randint(4, 512)
        args.batch_size = np.random.randint(1, args.pool_size // 2)
        batch_size = args.batch_size
        
        pool = train_pool(
            args,
            seed_graph,
            target_graph
        )
        
        seed_hidden = pool.seed_hidden.clone().detach().cpu()
        batch_data = pool.get_batch(batch_size)
        ids = batch_data['ids']
        batch_coords = batch_data['coords']
        batch_hidden = batch_data['hidden']
        
        # perterb coords + hidden
        batch_coords = batch_coords + torch.rand_like(batch_coords)
        batch_hidden = batch_hidden + torch.rand_like(batch_hidden)
        
        data = dict()
        data['ids'] = ids
        data['coords'] = batch_coords
        data['hidden'] = batch_hidden
        pool.update(data, 1, np.array([np.inf]*batch_size))
        
        clean_ids = []
        for id in ids:
            assert not torch.equal(pool.cache['coords'][id], seed_coords)
            assert not torch.equal(pool.cache['hidden'][id], seed_hidden)
            
            # * generate list of clean ids
            for _ in range(args.pool_size):
                id = (id + 1) % args.pool_size
                if id not in ids and id not in clean_ids:
                    break
            clean_ids.append(id)
            
        for id in clean_ids:
            assert torch.equal(pool.cache['coords'][id], seed_coords)
            assert torch.equal(pool.cache['hidden'][id], seed_hidden)
            
        if (verbose): print(f'[pool.py] test {i} complete')
            
    if (verbose): print(f'[pool.py] finished running {num_tests} tests -- all succeeded')