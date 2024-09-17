import plotly.graph_objs as go
import numpy as np
import torch

def create_ploty_figure(coords: torch.Tensor, edges: torch.Tensor = None):
    x, y, z = np.split(coords.cpu().numpy(), 3, 1)
    coords_plot_data = go.Scatter3d(
        x=x.squeeze(1).tolist(),
        y=y.squeeze(1).tolist(),
        z=z.squeeze(1).tolist(),
        mode='markers',
        marker={'size': 4, 'opacity': 0.8}
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
            marker={'size': 4, 'opacity': 0.8}
        )
        data = [coords_plot_data, edges_plot_data]
    layout = go.Layout(margin={'l': 0, 'r': 0, 'b': 0, 't': 0})
    return go.Figure(data=data, layout=layout)