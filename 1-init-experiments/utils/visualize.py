from typing import List, Tuple
import plotly.graph_objs as go
import numpy as np
import torch

# * color palette
# * https://coolors.co/006e90-adc698-ffac81-5f1a37-30011e
colors_list = [
    'rgb(  0,   110,    144)',
    'rgb( 95,    26,     55)',
    'rgb(  4,   167,    119)',
    'rgb(245,   213,     71)',
    'rgb(207,    92,     54)',
]

def create_ploty_figure(
    coords: torch.Tensor, 
    edges: torch.Tensor = None,
    return_ploty_data: bool = False,
    transparency: float = 0.5,
    coords_color: str = colors_list[0],
    edges_color: str = colors_list[0],
):
    x, y, z = np.split(coords.cpu().numpy(), 3, 1)
    coords_plot_data = go.Scatter3d(
        x=x.squeeze(1).tolist(),
        y=y.squeeze(1).tolist(),
        z=z.squeeze(1).tolist(),
        mode='markers',
        marker={'size': 4, 'opacity': transparency, 'color': coords_color}
    )
    data = [coords_plot_data]
    if edges is not None:
        edges_sorted, _ = torch.sort(edges.permute([1, 0]), dim=1)
        unique_pairs = set([tuple(pair.tolist()) for pair in edges_sorted])
        edges_pruned = torch.tensor(list(unique_pairs))
        num = edges_pruned.shape[0]
        edges_pos = torch.zeros([num*2, 3])
        for i in range(num):
            edges_pos[i*2] = coords[int(edges_pruned[i, 0])]
            edges_pos[i*2+1] = coords[int(edges_pruned[i, 1])]
        edges_np = edges_pos.cpu().numpy()
        edges_np = np.insert(edges_np, np.arange(2, edges_np.shape[0], 2), [None, None, None], axis=0)
        x, y, z = np.split(edges_np, 3, 1)
        edges_plot_data = go.Scatter3d(
            x=x.squeeze(1).tolist(),
            y=y.squeeze(1).tolist(),
            z=z.squeeze(1).tolist(),
            mode='lines',
            marker={'size': 4, 'opacity': transparency, 'color': edges_color}
        )
        data = [coords_plot_data, edges_plot_data]
        
    if return_ploty_data:
        return data
    else:
        layout = go.Layout(margin={'l': 0, 'r': 0, 'b': 0, 't': 0})
        return go.Figure(data=data, layout=layout)

def plot_multiple_graphs(
    graphs: List[Tuple],
    # unique_colors: bool = True
):
    all_data = []
    for i, graph in enumerate(graphs):
        coords, edges = graph
        coords_data, edges_data = create_ploty_figure(
            coords, 
            edges, 
            return_ploty_data=True,
            coords_color = colors_list[i],
            edges_color = colors_list[i],
        )
        all_data.append(coords_data)
        all_data.append(edges_data)
        
    layout = go.Layout(margin={'l': 0, 'r': 0, 'b': 0, 't': 0})
    return go.Figure(data=all_data, layout=layout)

# * view graph evolution
def evolve():
    pass