from utils.utils import furthest_point_dist
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
def create_evolve_figure(
    model: torch.nn.Module,
    num_steps: int = 50,
    frame_duration: int = 100,
    show_edges: bool = True,
    title: str = 'GNCA Evolution',
    coord_size: int = 4,
    coord_opactity: float = 0.5
):
    model.eval()
    _, \
    edges, _, \
    collection = model.run_for(num_steps, collect_all=True)
    
    def create_coords_dict(
        coords: torch.Tensor,
        name: str = 'coords',
        size: int = 4,
        color: str = colors_list[0],
        opacity: float = 0.5,
    ):
        x, y, z = np.split(coords.cpu().numpy(), 3, 1)
        coords_dict = {
            'type': 'scatter3d',
            'x': x.squeeze(1).tolist(),
            'y': y.squeeze(1).tolist(),
            'z': z.squeeze(1).tolist(),
            'mode': 'markers',
            'marker': {'size': size, 'opacity': opacity, 'color': color},
            'name': name
        }
        return coords_dict
    
    def create_edges_dict(
        edges: torch.Tensor,
        coords: torch.Tensor,
        name: str = 'edges',
        size: int = 4,
        color: str = colors_list[0],
        opacity: float = 0.5,
    ):
        edges = edges.clone()
        edges_sorted, _ = torch.sort(edges.permute([1, 0]), dim=1)
        unique_pairs = set([tuple(pair.tolist()) for pair in edges_sorted])
        edges_pruned = torch.tensor(list(unique_pairs))
        num = edges_pruned.shape[0]
        edges_pos = torch.zeros([num*2, 3])
        for j in range(num):
            edges_pos[j*2] = coords[int(edges_pruned[j, 0])]
            edges_pos[j*2+1] = coords[int(edges_pruned[j, 1])]
        edges_np = edges_pos.cpu().numpy()
        edges_np = np.insert(edges_np, np.arange(2, edges_np.shape[0], 2), [None, None, None], axis=0)
        x, y, z = np.split(edges_np, 3, 1)
        edges_dict = {
            'type': 'scatter3d',
            'x': x.squeeze(1).tolist(),
            'y': y.squeeze(1).tolist(),
            'z': z.squeeze(1).tolist(),
            'mode': 'lines',
            'marker': {'size': size, 'opacity': opacity, 'color': color},
            'name': name
        }
        return edges_dict

    # * create target coords/edges dictionaries
    target_coords_dict = create_coords_dict(
        model.target_coords, 
        'target-coords',
        size=coord_size,
        opacity=coord_opactity
    )
    if show_edges:
        target_edges_dict = create_edges_dict(
            edges, 
            model.target_coords, 
            'target-edges', 
            size=coord_size,
            opacity=coord_opactity
        )

    # * create frames of predicted coords/edges
    frames = []
    init_data = []
    for i, coords in enumerate(collection):
        pred_coords_dict = create_coords_dict(
            coords, 
            'pred-coords', 
            color=colors_list[4], 
            size=coord_size,
            opacity=coord_opactity
        )
        if show_edges:
            pred_edges_dict = create_edges_dict(
                edges, 
                coords,
                'pred-edges', 
                color=colors_list[4], 
                size=coord_size,
                opacity=coord_opactity
            )
        
        # * make data list
        data = None
        if show_edges: data = [pred_coords_dict, pred_edges_dict, target_coords_dict, target_edges_dict]
        else: data = [pred_coords_dict, target_coords_dict]
        
        # * save init data for figure
        if len(init_data) == 0:
            init_data = data
            
        # * create frame
        frames.append({
            'data': data,
            'name': f'step-{i}'
        })

    sliders_dict = {
        'active': 0,
        'yanchor': 'top',
        'xanchor': 'left',
        'currentvalue': {
            'font': {'size': 20},
            'prefix': 'Step:',
            'visible': True,
            'xanchor': 'right'
        },
        'transition': {'duration': frame_duration},
        'pad': {'b': 10, 't': 50},
        'len': 0.9,
        'x': 0.1,
        'y': 0,
        'steps': []
    }
    for i in range(num_steps+1):
        slider_step = {
            'args': [[f'step-{i}'], {
                'frame': {'duration': frame_duration, 'redraw': True},
                'mode': 'immediate',
                'transition': {'duration': frame_duration}
            }],
            'label': str(i),
            'method': 'animate'}
        sliders_dict['steps'].append(slider_step)

    updatemenus_dict = {
        'buttons': [
            {'args':
                [None, {'frame': 
                    {'duration': frame_duration, 'redraw': True},
                    'fromcurrent': True,
                    'transition': {'duration': frame_duration}}],
                    'label': 'Play',
                    'method': 'animate'
            }, 
            {'args': 
                [[None], {'frame': 
                    {'duration': 0, 'redraw': True},
                    'mode': 'immediate',
                    'transition': {'duration': 0}}],
                    'label': 'Pause',
                    'method': 'animate'
            }
        ],
        'x': 0.1,
        'y': 0,
        'pad': {'r': 10, 't': 87},
        'type': 'buttons',
        'xanchor': 'right',
        'yanchor': 'top',
        'direction': 'left',
        'showactive': False,
    }

    seed, _ = model.pool.seed
    _, d = furthest_point_dist(seed.cpu().numpy())
    layout = {
        'title': title,
        'scene': {
            'xaxis': {'range': [-d, d], 'autorange': False},
            'yaxis': {'range': [-d, d], 'autorange': False},
            'zaxis': {'range': [-d, d], 'autorange': False},
            'camera': dict(eye=dict(x=0.5, y=0.5, z=0.5))
        },
        'sliders': [sliders_dict],
        'updatemenus': [updatemenus_dict],
        'scene_aspectmode': 'cube',
        'margin': {'l': 0, 'r': 0, 'b': 0, 't': 35},
    }

    fig_dict = {
        'data': init_data,
        'layout': layout,
        'frames': frames
    }
    
    return go.Figure(fig_dict)