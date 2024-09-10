import torch
import colorsys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Optional, Union

def coord2scatter(
    coord: Union[torch.Tensor, np.ndarray],
    node_size: Optional[int] = 5,
    box_dim: Optional[int] = None,
    title: Optional[str] = '',
    show_ax_labels: Optional[bool] = False,
    show_ax_values: Optional[bool] = True,
    show_grid: Optional[bool] = False,
    zero_center: Optional[bool] = True,
    transparent: Optional[bool] = False,
    font_size: Optional[int] = 15,
    ax: Optional[plt.Axes] = None
):
    assert coord.ndim == 2 or coord.ndim == 3
    plt.rcParams.update({'font.size': font_size})

    if isinstance(coord, torch.Tensor):
        coord = coord.detach().cpu().numpy()
    coord = coord - coord.mean(0, keepdims=True) if zero_center else coord
    if coord.shape[1] == 2:
        if ax is None:
            fig, ax = plt.subplots()
            fig.tight_layout()
        ax.set_title(title)
        ax.scatter(coord[:, 0], coord[:, 1], s=node_size)
        if box_dim is not None:
            ax.set_xlim([-box_dim, box_dim])
            ax.set_ylim([-box_dim, box_dim])
        if show_grid:
            ax.grid(True)
        if show_ax_labels:
            ax.set_xlabel('x')
            ax.set_ylabel('y')
        if not show_ax_values:
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
        if transparent:
            ax.axis('off')
    else:
        if ax is None:
            fig = plt.figure()
            fig.tight_layout()
            ax = fig.add_subplot(projection='3d')
        ax.set_title(title)
        ax.scatter(coord[:, 0], coord[:, 1], coord[:, 2], s=node_size)
        if box_dim is not None:
            ax.axes.set_xlim3d(left=-box_dim, right=box_dim)
            ax.axes.set_ylim3d(bottom=-box_dim, top=box_dim)
            ax.axes.set_zlim3d(bottom=-box_dim, top=box_dim)
        if not show_grid:
            ax.grid(False)
        if show_ax_labels:
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
        if not show_ax_values:
            for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
                dim.set_ticks([])
        if transparent:
            ax.grid(False)
            for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
                dim.set_pane_color((1, 1, 1, 0))
            ax._axis3don = False

def get_colors(
    num_colors: int,
    seed: int = 42
):
    rand_state = np.random.RandomState(seed)
    colors = []
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i / 360.
        lightness = (30 + rand_state.rand() * 70) / 100.0
        saturation = (30 + rand_state.rand() * 70) / 100.0
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return colors

def plot_nx_graph_3d(
    nx_graph: nx.Graph,
    coord: np.ndarray,
    box_dim: Optional[float] = None,
    node_size: Optional[int] = 100,
    angle: Optional[float] = 30.0,
    zero_center: Optional[bool] = True,
    show_ax_values: Optional[bool] = False,
    show_ax_labels: Optional[bool] = False,
    show_grid: Optional[bool] = False,
    transparent: Optional[bool] = False,
    title: Optional[str] = '',
    unique_colors: Optional[bool] = False,
    ax: Optional[plt.Axes] = None
):
    assert coord.ndim == 2 and coord.shape[-1] == 3

    if ax is None:
        fig = plt.figure()
        fig.tight_layout()
        ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)
    ax.view_init(angle, angle)

    coord = coord - coord.mean(0, keepdims=True) if zero_center else coord
    node_xyz = np.array([coord[v] for v in sorted(nx_graph)])
    edge_xyz = np.array([(coord[u], coord[v]) for u, v in nx_graph.edges()])
    ax.scatter(*node_xyz.T, s=node_size, ec='w', c=get_colors(nx_graph.number_of_nodes()) if unique_colors else 'C0')
    for edge in edge_xyz:
        ax.plot(*edge.T, color='black')

    if box_dim is not None:
        ax.axes.set_xlim3d(left=-box_dim, right=box_dim)
        ax.axes.set_ylim3d(bottom=-box_dim, top=box_dim)
        ax.axes.set_zlim3d(bottom=-box_dim, top=box_dim)
    if show_ax_labels:
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
    if not show_grid:
        ax.grid(False)
    else:
        ax.grid(True)
    if not show_ax_values:
        from matplotlib.ticker import NullFormatter
        for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
            # dim.set_ticks([])
            dim.set_major_formatter(NullFormatter())
    if transparent:
        ax.grid(False)
        for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
            dim.set_pane_color((1, 1, 1, 0))
        ax._axis3don = False


def plot_traj_step_3d(
    coord_traj: Union[np.ndarray, torch.Tensor],
    t: int,
    title: Optional[str] = None,
    box_dim: Optional[float] = None,
    tail_len: Optional[int] = None,
    show_ax_labels: Optional[bool] = False,
    show_ax_values: Optional[bool] = True,
    show_grid: Optional[bool] = False,
    transparent: Optional[bool] = False,
    ax: Optional[plt.Axes] = None,
    font_size: Optional[int] = 15
):
    assert coord_traj.ndim == 3, '(steps, num nodes, coord dim)'
    plt.rcParams.update({'font.size': font_size})
    if isinstance(coord_traj, torch.Tensor):
        coord_traj = coord_traj.detach().cpu().numpy()
    if ax is None:
        fig = plt.figure(dpi=100)
        ax = fig.add_subplot(projection='3d')
    if title is not None:
        ax.set_title(title)
    if box_dim is not None:
        ax.set_xlim(-box_dim, box_dim)
        ax.set_ylim(-box_dim, box_dim)
        ax.set_zlim(-box_dim, box_dim)
    ax.scatter(coord_traj[t, :, 0], coord_traj[t, :, 1], coord_traj[t, :, 2], s=10, marker='o')
    if tail_len is not None:
        for j in range(coord_traj.shape[1]):
            x_tail = coord_traj[max(t - tail_len, 0): t + 1, j, 0]
            y_tail = coord_traj[max(t - tail_len, 0): t + 1, j, 1]
            z_tail = coord_traj[max(t - tail_len, 0): t + 1, j, 2]
            ax.plot(x_tail, y_tail, z_tail)
    if not show_grid:
        ax.grid(False)
    if show_ax_labels:
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
    if not show_ax_values:
        for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
            dim.set_ticks([])
    if transparent:
        ax.grid(False)
        for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
            dim.set_pane_color((1, 1, 1, 0))
        ax._axis3don = False

def edge_index2nx_graph(
    edge_index: torch.LongTensor,
    num_nodes: Optional[int] = None
):
    nx_graph = nx.Graph()
    for i in range(0 if num_nodes is None else num_nodes):
        nx_graph.add_node(i)
    for edge in edge_index.T:
        nx_graph.add_edge(edge[0].item(), edge[1].item())
    return nx_graph

def plot_edge_index(
    edge_index: torch.LongTensor,
    coord: Optional[Union[torch.Tensor, np.ndarray]] = None,
    num_nodes: Optional[int] = None,
    node_size: Optional[int] = 50,
    return_coord: Optional[bool] = False,
    with_labels: Optional[bool] = False,
    show_ax_labels: Optional[bool] = False,
    show_ax_values: Optional[bool] = False,
    show_grid: Optional[bool] = False,
    node_color: Optional[str] = 'C0',
    transparent: Optional[bool] = True,
    box_dim: Optional[int] = None,
    font_size: Optional[int] = 15,
    title: Optional[str] = '',
    ax: Optional[plt.Axes] = None
):
    plt.rcParams.update({'font.size': font_size})
    nx_graph = edge_index2nx_graph(edge_index, num_nodes)
    if coord is None:
        coord_dict = nx.spring_layout(nx_graph)
        coord = np.vstack([coord_dict[key] for key in coord_dict])
    assert coord.shape[-1] == 2 or coord.shape[-1] == 3
    if isinstance(coord, torch.Tensor):
        coord = coord.detach().cpu().numpy()
    if coord.shape[-1] == 2:
        if ax is None:
            fig, ax = plt.subplots()
        ax.set_title(title)
        nx.draw(nx_graph, coord, with_labels=with_labels, node_size=node_size, node_color=node_color, ax=ax)
        if box_dim is not None:
            ax.axes.set_xlim([-box_dim, box_dim])
            ax.axes.set_ylim([-box_dim, box_dim])
        if show_ax_values:
            ax.set_axis_on()
            ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    else:
        plot_nx_graph_3d(
            edge_index2nx_graph(edge_index), coord, node_size=node_size, box_dim=box_dim, title=title, ax=ax,
            show_ax_values=show_ax_values, show_grid=show_grid, show_ax_labels=show_ax_labels, transparent=transparent)
    if return_coord:
        return coord