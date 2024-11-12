import cv2
import numpy as np
import maxflow
from examples_utils import plot_graph_2d


def create_graph(dim=(5, 5)):
    g = maxflow.Graph[float]()
    nodeids = g.add_grid_nodes(dim)

    weights = np.array([[100, 110, 120, 130, 140]]).T + \
        np.array([0, 2, 4, 6, 8])

   # Edges pointing left
    structure = np.zeros((3, 3))
    structure[1, 0] = 1
    g.add_grid_edges(nodeids, structure=structure,
                     weights=weights, symmetric=False)

    # Edges pointing right
    structure = np.zeros((3, 3))
    structure[1, 2] = 1
    g.add_grid_edges(nodeids, structure=structure,
                     weights=weights, symmetric=False)

    # Edges pointing up
    structure = np.zeros((3, 3))
    structure[0, 1] = 1
    g.add_grid_edges(nodeids, structure=structure,
                     weights=weights, symmetric=False)

    # Edges pointing down
    structure = np.zeros((3, 3))
    structure[2, 1] = 1
    g.add_grid_edges(nodeids, structure=structure,
                     weights=weights, symmetric=False)

    # Source node connected to leftmost non-terminal nodes.
    left = nodeids[:, :]
    g.add_grid_tedges(left, np.inf, 0)
    # Sink node connected to rightmost non-terminal nodes.
    right = nodeids[:, :]
    g.add_grid_tedges(right, 0, np.inf)

    return nodeids, g


nodeids, g = create_graph((5, 5))

plot_graph_2d(g, nodeids.shape)

g.maxflow()
print(g.get_grid_segments(nodeids))
