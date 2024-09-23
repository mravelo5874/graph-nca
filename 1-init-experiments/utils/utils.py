from sklearn import neighbors
import numpy as np

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