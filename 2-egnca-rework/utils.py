from data.generate import \
    generate_line_graph, \
    generate_square_plane_graph, \
    retrieve_bunny_graph, \
    generate_cube_graph
from graph import graph
import numpy as np
import torch

def default_namespace():
    import datetime
    now = datetime.datetime.now()
    date_time = now.strftime('%Y-%m-%d@%H-%M-%S')

    from argparse import Namespace
    args = Namespace()
    
    # model parameters
    args.device = 'cuda'
    args.message_dim = 32
    args.hidden_dim = 16
    args.has_attention = True
    
    # training parameters
    args.epochs = 10_000
    args.pool_size = 256
    args.batch_size = 4
    args.start_lr = 0.0005
    args.end_lr = 0.00001
    args.beta1 = 0.9
    args.beta2 = 0.999
    args.wd = 1e-5
    args.factor_sch = 0.5
    args.patience_sch = 500
    args.min_steps = 15
    args.max_steps = 25
    args.log_rate = 100
    args.comp_edge_percent = 0.5 # random percent of ALL potential edges to use for loss calculations
    
    # graph parameters
    args.graph = 'line'
    args.size = 4
    args.length = 1.0
    args.seed_std = 0.5
    args.init_hidden = 'ones' # 'rand'
    
    # file 
    args.file_name = f'{args.graph}{args.size}-{date_time}'
    args.save_to = 'logs'
    return args

def print_batch_dict(
    batch_dict: dict
):
    ids = batch_dict['ids']
    coords = batch_dict['coords']
    hidden = batch_dict['hidden']
    comp_edges = batch_dict['comp_edges']
    comp_lens = batch_dict['comp_lens']
    print (f'ids.shape: {ids.shape}')
    print (f'coords.shape: {coords.shape}')
    print (f'hidden.shape: {hidden.shape}')
    print (f'comp_edges.shape: {comp_edges.shape}')
    print (f'comp_lens.shape: {comp_lens.shape}')
    
def expand_edge_tensor(
    edges: torch.LongTensor,
    n_nodes: int,
    size: int, # generally, this is the batch_size
) -> torch.Tensor:
    
    edges_list = []
    for i in range(size):
        batch_edges = edges.detach().clone().add(i*n_nodes)
        edges_list.append(batch_edges)
    edges = torch.cat(edges_list, dim=1).long()
    return edges

def create_graph(
    graph_name: str, 
    size: int,
    length: float
) -> graph:
    
    if graph_name == 'line':
        coords, edges = generate_line_graph(size, length)
    elif graph_name == 'grid':
        coords, edges = generate_square_plane_graph(size, length)
    elif graph_name == 'cube':
        coords, edges = generate_cube_graph(size, length)
    elif graph_name == 'bunny':
        coords, edges = retrieve_bunny_graph()
    else:
        print (f'[utils.py] error! invalid graph name used: {graph_name} -- returning None')
        return None
    return graph(coords, edges)

def furthest_point_dist(
    point_cloud: np.ndarray[float],
    center: np.ndarray[float] = np.array([0.0, 0.0, 0.0])
):
    max_distance = 0
    furthest_point = None
    assert point_cloud.shape[1] == 3

    for i in range(point_cloud.shape[0]):
        point = point_cloud[i]
        distance = np.linalg.norm(point-center)  # Calculate Euclidean distance
        if distance > max_distance:
            max_distance = distance
            furthest_point = point

    return furthest_point, max_distance

from trainer import trainer
def compare_pool_vs_runfor_graphs(
    trainer: trainer
):
    from visualize import create_ploty_figure_multiple, rgba_colors_list
    from IPython.display import clear_output
    from plotly.subplots import make_subplots
    import plotly
    
    clear_output()
    trgt_coords, edges = trainer.target_graph.get()
    graph_data = trainer.pool.get_random_graph()
    index = graph_data['id']
    pred_coords = graph_data['coords']
    steps = graph_data['steps']
    loss = graph_data['loss']
    seed_coords, _ = trainer.pool.seed_graph.get()
    seed_color = rgba_colors_list[4]
    seed_color[3] = 0
    plotly.offline.init_notebook_mode()
    fig1 = create_ploty_figure_multiple(
        graphs=[(trgt_coords, edges),
                (pred_coords, edges),
                (seed_coords, edges)],
        coords_color=[rgba_colors_list[0], rgba_colors_list[1], seed_color],
        edges_color=[rgba_colors_list[0], rgba_colors_list[1], seed_color]
    )

    # * show "runfor" graph
    runfor_data = trainer.runfor(steps)
    runfor_coords = runfor_data['coords']
    fig2 = create_ploty_figure_multiple(
        graphs=[(trgt_coords, edges),
                (runfor_coords, edges),
                (seed_coords, edges)],
        coords_color=[rgba_colors_list[0], rgba_colors_list[1], seed_color],
        edges_color=[rgba_colors_list[0], rgba_colors_list[1], seed_color]
    )
    
    # * combine figures into one
    runfor_loss = runfor_data['loss']
    fig3 = make_subplots(
        rows=1, cols=2, 
        vertical_spacing=0.02, 
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=[
            f'pool graph #{index}, steps: {steps}, loss: {loss}',
            f'run_for() graph, steps: {steps} loss: {runfor_loss}'
    ])
    for j in fig1.data:
        fig3.add_trace(j, row=1, col=1)
    for j in fig2.data:
        fig3.add_trace(j, row=1, col=2)
    plotly.offline.iplot(fig3)