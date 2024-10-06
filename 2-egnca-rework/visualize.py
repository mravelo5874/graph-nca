from utils import furthest_point_dist
from typing import List, Tuple
import plotly.graph_objs as go
import numpy as np
import torch

# * color palette
# * https://coolors.co/006e90-adc698-ffac81-5f1a37-30011e
rgba_colors_list = [
    [0,     110,    144,    255],
    [95,    26,     55,     255],
    [4,     167,    119,    255],
    [245,   213,    71,     255],
    [207,   92,     4,      255],
]

def create_ploty_figure(
    coords: torch.Tensor, 
    edges: torch.Tensor = None,
    return_ploty_data: bool = False,
    title: str = 'My Graph',
    coords_color: List[int] = rgba_colors_list[0],
    coord_size: int = 2,
    show_edges: bool = True,
    edges_color: List[int] = rgba_colors_list[0],
    edges_size: int = 2,
):
    assert len(coords_color) == 4
    assert len(edges_color) == 4
    x, y, z = np.split(coords.cpu().numpy(), 3, 1)
    coords_plot_data = go.Scatter3d(
        x=x.squeeze(1).tolist(),
        y=y.squeeze(1).tolist(),
        z=z.squeeze(1).tolist(),
        mode='markers',
        marker={
            'size': coord_size, 
            'opacity': float(coords_color[3])/255.0, 
            'color': f'rgb({coords_color[0]}, {coords_color[1]}, {coords_color[2]})'
        }
    )
    data = [coords_plot_data]
    if show_edges and edges is not None:
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
            line={
                'width': edges_size,
                'color': f'rgba({edges_color[0]}, {edges_color[1]}, {edges_color[2]}, {float(edges_color[3])/255.0})'
            }
        )
        data = [coords_plot_data, edges_plot_data]
        
    if return_ploty_data:
        return data
    else:
        layout = go.Layout(
            margin={'l': 0, 'r': 0, 'b': 0, 't': 35},
            scene={'camera': dict(eye=dict(x=0.5, y=0.5, z=0.5))},
            title=title
        )
        return go.Figure(data=data, layout=layout)

def create_ploty_figure_multiple(
    graphs: List[Tuple[torch.Tensor, torch.Tensor]],
    title: str = 'Plotly Graph',
    coords_color: List[List[int]] = None,
    coord_size: List[int] = None,
    show_edges: bool = True,
    edges_color: List[List[int]] = None,
    edges_size: List[int] = None,
):
    # * set defaults
    n = len(graphs)
    if coords_color is not None: assert len(coords_color) == n
    else: coords_color = [rgba_colors_list[0]] * n
    if coord_size is not None: assert len(coord_size) == n
    else: coord_size = [2] * n
    if edges_color is not None: assert len(edges_color) == n
    else: edges_color = [rgba_colors_list[0]] * n
    if edges_size is not None: assert len(edges_size) == n
    else: edges_size = [2] * n
    
    all_data = []
    for i, graph in enumerate(graphs):
        coords, edges = graph
        coords_data, edges_data = create_ploty_figure(
            coords, 
            edges, 
            return_ploty_data=True,
            coords_color=coords_color[i],
            coord_size=coord_size[i],
            show_edges=show_edges,
            edges_color=edges_color[i],
            edges_size=edges_size[i],
        )
        all_data.append(coords_data)
        all_data.append(edges_data)
        
    layout = go.Layout(
        margin={'l': 0, 'r': 0, 'b': 0, 't': 35},
        title=title
    )
    return go.Figure(data=all_data, layout=layout)


def create_evolve_figure(
    model: torch.nn.Module,
    num_steps: int = 50,
    frame_duration: int = 100,
    title: str = 'Plotly Graph',
    target_coords_color: List[int] = rgba_colors_list[0],
    target_coord_size: int = 2,
    pred_coords_color: List[int] = rgba_colors_list[4],
    pred_coord_size: int = 2,
    show_edges: bool = True,
    target_edges_color: List[int] = rgba_colors_list[0],
    target_edges_size: int = 2,
    pred_edges_color: List[int] = rgba_colors_list[4],
    pred_edges_size: int = 2,
):
    assert len(target_coords_color) == 4
    assert len(pred_coords_color) == 4
    assert len(target_edges_color) == 4
    assert len(pred_edges_color) == 4
    model.eval()
    _, \
    edges, _, \
    collection = model.run_for(num_steps, collect_all=True)
    
    def create_coords_dict(
        coords: torch.Tensor,
        name: str,
        size: int,
        color: List[int],
    ):
        x, y, z = np.split(coords.cpu().numpy(), 3, 1)
        coords_dict = {
            'name': name,
            'type': 'scatter3d',
            'x': x.squeeze(1).tolist(),
            'y': y.squeeze(1).tolist(),
            'z': z.squeeze(1).tolist(),
            'mode': 'markers',
            'marker': {
                'size': size, 
                'opacity': float(color[3])/255.0,
                'color': f'rgb({color[0]}, {color[1]}, {color[2]})'
            }
        }
        return coords_dict
    
    def create_edges_dict(
        edges: torch.Tensor,
        coords: torch.Tensor,
        name: str,
        size: int,
        color: List[int],
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
            'name': name,
            'type': 'scatter3d',
            'x': x.squeeze(1).tolist(),
            'y': y.squeeze(1).tolist(),
            'z': z.squeeze(1).tolist(),
            'mode': 'lines',
            'line': {
                'width': size, 
                'color': f'rgba({color[0]}, {color[1]}, {color[2]}, {float(color[3])/255.0})'
            }
        }
        return edges_dict

    # * create target coords/edges dictionaries
    target_coords_dict = create_coords_dict(
        model.target_coords, 
        'target-coords',
        color=target_coords_color,
        size=target_coord_size,
    )
    if show_edges:
        target_edges_dict = create_edges_dict(
            edges, 
            model.target_coords, 
            'target-edges',
            color=target_edges_color,
            size=target_edges_size,
        )

    # * create frames of predicted coords/edges
    frames = []
    init_data = []
    for i, coords in enumerate(collection):
        pred_coords_dict = create_coords_dict(
            coords, 
            'pred-coords', 
            color=pred_coords_color, 
            size=pred_coord_size,
        )
        if show_edges:
            pred_edges_dict = create_edges_dict(
                edges, 
                coords,
                'pred-edges', 
                color=pred_edges_color,
                size=pred_edges_size,
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
            'camera': dict(eye=dict(x=1, y=1, z=1))
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