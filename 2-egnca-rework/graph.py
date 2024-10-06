import torch

class graph():
    def __init__(
        self,
        coords: torch.Tensor,
        edges: torch.LongTensor
    ):
        self.coords = coords.clone().detach().cpu()
        self.edges = edges.clone().detach().cpu()
        
    def __reduce__(self):
        return (self.__class__, (self.coords, self.edges))
        
    def get(self):
        coords = self.coords.clone().detach().cpu()
        edges = self.edges.clone().detach().cpu()
        return coords, edges
    
    def get_np(self):
        coords, edges = self.get()
        return coords.numpy(), edges.numpy()