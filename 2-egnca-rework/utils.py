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
    
import torch
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
    
from graph import graph
from data.generate import \
    generate_line_graph, \
    generate_square_plane_graph, \
    retrieve_bunny_graph, \
    generate_cube_graph

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