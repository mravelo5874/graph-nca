import torch

class graph():
    def __init__(
        self,
        coords: torch.Tensor,
        edges: torch.LongTensor,
        hidden: torch.Tensor = None
    ):
        self.coords = coords.clone().detach().cpu()
        self.edges = edges.clone().detach().cpu()
        self.hidden = None
        if hidden is not None: 
            self.hidden = hidden.clone().detach().cpu()
        
    def __reduce__(self):
        return (self.__class__, (self.coords, self.edges, self.hidden))
        
    def get(self):
        coords = self.coords.clone().detach().cpu()
        edges = self.edges.clone().detach().cpu()
        hidden = None
        if self.hidden is not None: 
            hidden = self.hidden.clone().detach().cpu()
        return coords, edges, hidden
    
    def get_np(self):
        coords, edges, hidden = self.get()
        if hidden is not None:
            hidden = hidden.numpy()
        return coords.numpy(), edges.numpy(), hidden