from torch_geometric.data import Data, Dataset
from typing import Optional
import torch

class GraphDataset(Dataset):
    
    def __init__(
        self,
        coords: torch.Tensor,
        edges: torch.LongTensor,
        scale: Optional[float] = 1.0,
        length: Optional[int] = 1,
    ):
        super().__init__()
        assert length > 0

        self.coords = coords * scale
        self.edges = edges
        self.scale = scale
        self.length = length
        self.num_nodes = self.coord.size(0)
        
    def get(
        self,
        index: int
    ):
        data = Data()
        return data
    
    def len(self):
        return self.length