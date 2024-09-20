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
        seed_coords = None
        seed_hidden = None
        
        self.egnn = egnn([
            egc(args.coords_dim, args.hidden_sim, args.message_dim, has_attention=True),
        ]).to(args.device)
        
        self.pool = TrainPool(
            pool_size=args.pool_size,
            seed_coords=seed_coords,
            seed_hidden=seed_hidden,
        )
        
    def forward(
        self,
    ):
        pass
        
    def train(
        self,
        
    ):
        for i in range(self.args.epochs+1):
            batch_ids, batch_coords, batch_hidden = self.pool.get_batch(self.args.batch_size)
            batch_coords, batch_hidden = self.egnn()
            self.pool.update_pool(batch_ids, batch_coords, batch_hidden)