from argparse import Namespace
from data.generate import \
    generate_line_graph, \
    generate_square_plane_graph, \
    retrieve_bunny_graph, \
    generate_cube_graph, \
    generate_geodesic_polyhedron_graph
from graph import graph
import numpy as np
import datetime
import torch

def default_namespace():
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
    args.save_to = 'logs'
    
    # graph parameters
    args.graph = 'line'
    args.size = 4
    args.length = 1.0
    args.seed_std = 0.5
    args.init_hidden = 'ones' # 'rand'
    
    # update file name and return
    args = update_namespace(args)
    return args

def update_namespace(args: Namespace):
    now = datetime.datetime.now()
    date_time = now.strftime('%Y-%m-%d@%H-%M-%S')
    args.file_name = f'{args.graph}{args.size}-{date_time}'
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
    elif graph_name == 'poly':
        coords, edges = generate_geodesic_polyhedron_graph(1, True)
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
    trgt_coords, edges, _ = trainer.target_graph.get()
    graph_data = trainer.pool.get_random_graph()
    index = graph_data['id']
    pred_coords = graph_data['coords']
    assert not torch.equal(pred_coords, trgt_coords)
    steps = graph_data['steps']
    loss = graph_data['loss']
    fig1 = create_ploty_figure_multiple(
        graphs=[(trgt_coords, edges),
                (pred_coords, edges)],
        coords_color=[rgba_colors_list[0], rgba_colors_list[1]],
        edges_color=[rgba_colors_list[0], rgba_colors_list[1]]
    )

    # * show "runfor" graph
    runfor_data = trainer.runfor(steps)
    runfor_coords = runfor_data['coords']
    fig2 = create_ploty_figure_multiple(
        graphs=[(trgt_coords, edges),
                (runfor_coords, edges)],
        coords_color=[rgba_colors_list[0], rgba_colors_list[1]],
        edges_color=[rgba_colors_list[0], rgba_colors_list[1]]
    )
    
    # * combine figures into one
    runfor_loss = runfor_data['loss']
    fig3 = make_subplots(
        rows=1, cols=2, 
        vertical_spacing=0.02, 
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=[
            f'pool graph #{index}, steps: {steps}, loss: {loss}',
            f'runfor graph, steps: {steps} loss: {runfor_loss}'
    ])
    for i in fig1.data: fig3.add_trace(i, row=1, col=1)
    for i in fig2.data: fig3.add_trace(i, row=1, col=2)
    plotly.offline.init_notebook_mode()
    plotly.offline.iplot(fig3)
    
def compare_collections(
    batch_collection: dict,
    graph_collection: dict,
    n_nodes: int,
    n_edges: int
):
    # coords_dif
    b = batch_collection['coords_dif'][0:n_edges]
    g = graph_collection['coords_dif']
    assert torch.equal(b, g)
    
    # coords_l2
    b = batch_collection['coords_l2'][0:n_edges]
    g = graph_collection['coords_l2']
    assert torch.equal(b, g)
    
    # h_i
    b = batch_collection['h_i'][0:n_edges]
    g = graph_collection['h_i']
    assert torch.equal(b, g)
    
    # h_j
    b = batch_collection['h_j'][0:n_edges]
    g = graph_collection['h_j']
    assert torch.equal(b, g)
    
    # message_mlp_input
    b = batch_collection['message_mlp_input'][0:n_edges]
    g = graph_collection['message_mlp_input']
    assert torch.equal(b, g)
    
    # m_ij
    b = batch_collection['m_ij'][0:n_edges]
    g = graph_collection['m_ij']
    d = b-g
    print (f'(dev) m_ij diff:\n{d}')
    assert torch.equal(b, g)
    
    # coord_mlp_out
    b = batch_collection['coord_mlp_out'][0:n_edges]
    g = graph_collection['coord_mlp_out']
    assert torch.equal(b, g)

    # coord_trans
    b = batch_collection['coord_trans'][0:n_edges]
    g = graph_collection['coord_trans']
    assert torch.equal(b, g)
    
    # coord_trans_matrix
    b = batch_collection['coord_trans_matrix'][0:n_nodes, 0:n_nodes]
    g = graph_collection['coord_trans_matrix']
    assert torch.equal(b, g)
    
    # node_i_nbors
    b = batch_collection['node_i_nbors'][0:n_nodes]
    g = graph_collection['node_i_nbors']
    assert torch.equal(b, g)
    
    # trans_sum_i
    b = batch_collection['trans_sum_i'][0:n_nodes]
    g = graph_collection['trans_sum_i']
    assert torch.equal(b, g)
    
    # coords_out
    b = batch_collection['coords_out'][0:n_nodes]
    g = graph_collection['coords_out']
    assert torch.equal(b, g)
    
    # m_ij_matrix
    b = batch_collection['m_ij_matrix'][0:n_nodes, 0:n_nodes]
    g = graph_collection['m_ij_matrix']
    assert torch.equal(b, g)
    
    # m_i
    b = batch_collection['m_i'][0:n_nodes]
    g = graph_collection['m_i']
    assert torch.equal(b, g)
    
    # hidden_mlp_input
    b = batch_collection['hidden_mlp_input'][0:n_nodes]
    g = graph_collection['hidden_mlp_input']
    assert torch.equal(b, g)
    
    # hidden_mlp_out
    b = batch_collection['hidden_mlp_out'][0:n_nodes]
    g = graph_collection['hidden_mlp_out']
    # d = b-g
    # print (f'hidden_mlp_out diff:\n{d}')
    assert torch.equal(b, g)
    
    # hidden_out
    b = batch_collection['hidden_out'][0:n_nodes]
    g = graph_collection['hidden_out']
    assert torch.equal(b, g)
    
    print ('batch and graph are equal (equal)!')