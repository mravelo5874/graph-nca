from typing import Optional
import numpy as np
import torch

class TrainPool:
    
    def __init__(
        self,
        pool_size: int,
        seed_coords: torch.Tensor,
        seed_hidden: torch.Tensor,
        device: Optional[str] = 'cuda'
    ):
        self.pool_size = pool_size
        self.device = device
        self.cache = dict()
        self.cache['coords'] = seed_coords.clone().repeat([pool_size, 1, 1])
        self.cache['hidden'] = seed_hidden.clone().repeat([pool_size, 1, 1])
        self.cache['iters'] = [0] * pool_size
    
    def get_batch(
        self,
        batch_size: int,
        replace_lowest_loss: bool = True
    ):
        batch_ids = np.random.choice(self.pool_size, batch_size, replace=False)
        batch_coords = self.cache['coords'][batch_ids]
        batch_hidden = self.cache['hidden'][batch_ids]
        
        # * re-order batch based on loss
        if replace_lowest_loss:
            pass
            # loss_ranks = torch.argsort(voxutil.voxel_wise_loss_function(x, target_ten, _dims=[-1, -2, -3, -4]), descending=True)
            # x = x[loss_ranks]
            # # * re-add seed into batch
            # x[:1] = seed_ten
            
        return batch_ids, batch_coords, batch_hidden
    
    def update_pool(
        self,
        batch_ids: np.ndarray[int],
        batch_coords: torch.Tensor,
        batch_hidden: torch.Tensor
    ):
        self.cache['coords'][batch_ids] = batch_coords
        self.cache['hidden'][batch_ids] = batch_hidden
        
def test_pool_functionality(
    num_tests: int = 100,
    verbose: bool = False
):
    
    from data.generate import generate_bunny_graph
    
    seed_coords, _ = generate_bunny_graph()
    seed_hidden = torch.rand([seed_coords.shape[0], 16])
    
    for i in range(num_tests):
        
        pool = TrainPool(
            pool_size=np.random.randint(1, 512),
            seed_coords=seed_coords,
            seed_hidden=seed_hidden
        )
        
        batch_size = np.random.randint(1, pool.pool_size // 2)
        batch_ids, batch_coords, batch_hidden = pool.get_batch(batch_size)
        batch_coords = batch_coords + torch.rand_like(batch_coords)
        batch_hidden = batch_hidden + torch.rand_like(batch_hidden)
        pool.update_pool(batch_ids, batch_coords, batch_hidden)
        
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
        
    