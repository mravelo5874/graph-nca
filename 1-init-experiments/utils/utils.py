import numpy as np
from sklearn import neighbors

def compute_connectivity(positions, radius, add_self_edges):
    """Get the indices of connected edges with radius connectivity.

    Args:
        positions: Positions of nodes in the graph. Shape:
        [num_nodes_in_graph, num_dims].
        radius: Radius of connectivity.
        add_self_edges: Whether to include self edges or not.

    Returns:
        senders indices [num_edges_in_graph]
        receiver indices [num_edges_in_graph]

    """
    tree = neighbors.KDTree(positions)
    receivers_list = tree.query_radius(positions, r=radius)
    num_nodes = len(positions)
    senders = np.repeat(range(num_nodes), [len(a) for a in receivers_list])
    receivers = np.concatenate(receivers_list, axis=0)

    if not add_self_edges:
        # Remove self edges.
        mask = senders != receivers
        senders = senders[mask]
        receivers = receivers[mask]

    return senders, receivers