from torch_geometric.nn import PairNorm
from argparse import Namespace
from egnn import egnn, egc
import torch
import json

class fixed_target_egnca(torch.nn.Module):
    
    def __init__(
        self,
        args: Namespace,
    ):
        super(fixed_target_egnca, self).__init__()
        self.args = args
        self.pairnorm = PairNorm(scale=1.0)
        self.egnn = egnn([egc(args)]).to(args.device)
        
    def save(
        self,
        dir: str,
        file_name: str,
        verbose: bool = False,
    ):
        torch.save(self.state_dict(), f'{dir}/{file_name}.pt')
        with open(f'{dir}/args.txt', 'w') as f:
            json.dump(self.args.__dict__, f, indent=2, default=lambda o: '<not serializable>')
        if verbose: print (f'[models.py] saved model to: {dir}/{file_name}')
        
    def forward(
        self,
        coords: torch.Tensor,
        hidden: torch.Tensor,
        edges: torch.LongTensor,
        collect: bool = False
    ):
        out_coords, \
        out_hidden, \
        collection = self.egnn(
            coords,
            hidden,
            edges,
            collect
        )
        out_hidden = self.pairnorm(out_hidden)
        return out_coords, out_hidden, collection