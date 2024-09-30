import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cvxpy as cp
import networkx as nx
import scipy

import graph_tool as gt
import numba
import numpy as np

from graph_tool.topology import shortest_distance
from numba.core import types
from typing import Optional, Union, Iterable



FLOAT = np.float64

def get_weights_matrix(graph: nx.DiGraph, key: str = "bandwidth") -> np.ndarray:
    """
    Extract capacity matrix from graph
    :param graph: nx.DiGraph,: graph with attribute cost on edges
    :param key: str, default="bandwidth", name of attribute to obtain weights matrix
    :return:
        capacity_matrix: ndarray of shape (num_nodes, num_nodes)
    """
    capacity_matrix = nx.adjacency_matrix(graph, weight=key)
    capacity_matrix = capacity_matrix.toarray()
    return capacity_matrix


def get_incidence_matrix(graph: nx.DiGraph) -> np.ndarray:
    """
    Construct incidence matrix
    :param graph: nx.DiGraph,: graph with attribute cost on edges
    :return:
        incidence_matrix: ndarray of shape (num_nodes, num_edges), incidence matrix
    """
    incidence_matrix = nx.incidence_matrix(graph, edgelist=graph.edges, oriented=True)
    incidence_matrix = incidence_matrix.toarray()
    return incidence_matrix


def get_weights_using_weights_matrix(graph: nx.DiGraph, key: str = "bandwidth") -> np.ndarray:
    """
    Construct extract all weights
    :param graph: nx.DiGraph,: graph with attribute cost on edges
    :param key: str, default="bandwidth", name of attribute to obtain weights matrix
    :return:
        capacities: ndarray of shape (num_nodes), all capacities in graph
    """
    weights_matrix = get_weights_matrix(graph, key=key)
    capacities = np.triu(weights_matrix)
    capacities = capacities[capacities != 0]
    return capacities


def get_weights(graph: nx.DiGraph, key: str) -> np.ndarray:
    """
    Extract edge weights
    :param graph: nx.DiGraph, graph with weights on edges
    :param key: str, name of attribute to obtain weights
    :return:
        weights: ndarray of shape (num_nodes), all edge weights in graph
    """
    return np.array(list(nx.get_edge_attributes(graph, key).values()), dtype=FLOAT)


def calculate_laplacian(graph: nx.DiGraph, lapl_type: str, key: str = "bandwidth") -> np.ndarray:
    """
    Calculate laplacian matrix: L := D - W, where D is diagonal matrix with d_{ii} = sum_j w_ij
    if lapl_type == 'in' else d_{ii} = sum_j w_ji
    :param graph: nx.DiGraph,: graph with attribute cost on edges
    :param key: str, default="bandwidth", name of attribute to obtain weights matrix
    :return:
        L: ndarray of shape (num_nodes, num_nodes), laplacian matrix for input weight matrix
    """
    weights_matrix = get_weights_matrix(graph, key=key)
    return calculate_laplacian_from_weights_matrix(weights_matrix, typ=lapl_type)


def calculate_laplacian_from_weights_matrix(weights_matrix: np.ndarray, typ: str) -> np.ndarray:
    """
    Calculate laplacian matrix: L := D - W, where D is diagonal matrix with out-degrees or in degrees
    depending on typ. If typ == 'out', then d_ii = sum_j w_ij; if typ == 'in', then d_ii = sum_j w_ji.
    :param weights_matrix: ndarray of shape (num_nodes, num_nodes), symmetric weights matrix of graph
    :param typ: str, 'out' to compute out-degree laplacian, 'in' to compute in-degree laplacian
    :return:
        L: ndarray of shape (num_nodes, num_nodes), laplacian matrix for input weight matrix
    """
    axis = 0 if typ == "in" else 1
    return np.diag(weights_matrix.sum(axis)) - weights_matrix


def get_var_value(var: Optional[cp.Variable]) -> Optional[float]:
    """
    Get cvxpy.Variable value if not None
    :param var: cvxpy.Variable, variable to extract value
    :return:
        value: var.value if var is not None, else None
    """
    return var.value if var is not None else None


def shortest_paths_lengths(gt_graph, weights):
    num_vertices = gt_graph.num_vertices()
    distances_matrix = np.zeros((num_vertices, num_vertices))

    targets = np.arange(num_vertices)
    # Проходимся по каждой вершине в графе в качестве источника
    for source in range(num_vertices):
        # Вычисляем кратчайшие расстояния от source до всех остальных вершин
        dist_arr = shortest_distance(gt_graph, source=source, target=targets, weights=weights, pred_map=False)
        distances_matrix[source] = dist_arr

    assert np.all(np.diag(distances_matrix) == 0)
    return distances_matrix


# def find_ones_indices(matrix):
#     indices = []
#     n = len(matrix)
#     for i in range(n):
#         for j in range(n):
#             if matrix[i][j] == 1:
#                 indices.append((i, j))
#     return indices


# TODO: use numba
def get_path_edges(gt_graph, pred_map, source, target):
    path = [target]
    while path[-1] != source:
        path.append(int(pred_map[path[-1]]))
    path_edges = []
    for i in range(len(path) - 1, 0, -1):
        edge = gt_graph.edge(path[i], path[i - 1])
        path_edges.append(edge)
    return path_edges


# нахожу номер элемента в матрице по его индекссу из flatten матрицы
def get_index(matrix_shape, flat_index):
    num_cols = matrix_shape[1]
    row_index = flat_index // num_cols
    col_index = flat_index % num_cols
    return [row_index, col_index]


# Функция для визуализации сходимости
def plot_convergence(current_flows, bandwidths, kirch, k, add_bw, lam, lam_delta):
    plt.figure(figsize=(15, 5))

    # График суммарного потока
    plt.subplot(1, 3, 1)
    plt.plot(current_flows, label="Current Flow")
    plt.plot(bandwidths.a, label="Bandwidth", linestyle="--")
    if add_bw is not None:
        plt.plot(add_bw, label="Added bandwidth", linestyle="dotted")
    plt.title("Flow vs Bandwidth")
    plt.ylabel("Flow")
    plt.legend()

    # График сохранения потока
    plt.subplot(1, 3, 2)
    plt.plot(kirch[:k], label="Flow Conservation")
    plt.title("Flow Conservation over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Mismatch")
    plt.yscale("log")
    plt.legend()

    # # График зазора
    # plt.subplot(1, 4, 3)
    # plt.plot(delta, label='delta')
    # plt.title('Delta')
    # plt.legend()

    # сходимость к лямбде
    plt.subplot(1, 3, 3)
    plt.plot(lam_delta, label="delta")
    plt.title("lambda's delta")
    plt.yscale("log")
    plt.legend()

    plt.tight_layout()
    plt.show()
    
    
    from typing import Iterable, Optional


@numba.njit
def sum_flows_from_tree(
    source: int, targets: np.ndarray, pred_map_arr: np.ndarray, traffic_mat: np.ndarray, edge_to_ind: numba.typed.Dict
) -> np.ndarray:
    num_edges = len(edge_to_ind)
    flows_e = np.zeros(num_edges)
    for v in targets:
        corr = traffic_mat[source, v]
        while v != source:
            v_pred = pred_map_arr[v]
            flows_e[edge_to_ind[(v_pred, v)]] += corr
            v = v_pred
    return flows_e


def flows_on_shortest_gt(
    graph: gt.Graph,
    traffic_mat: np.ndarray,
    weights: gt.EdgePropertyMap,
    sources: Optional[Iterable] = None,
    targets: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Returns flows on edges for each ij-pair in sources and targets (obtained from flows on shortest paths w.r.t costs induced by weights). If sources and/or targets are not specified, all nodes assumed"""
    num_nodes, num_edges = graph.num_vertices(), graph.num_edges()

    edges_arr = graph.get_edges()
    edge_to_ind = numba.typed.Dict.empty(key_type=types.UniTuple(types.int64, 2), value_type=numba.core.types.int64)
    for i, edge in enumerate(edges_arr):
        edge_to_ind[tuple(edge)] = i

    flows_on_shortest_e = np.zeros(num_edges)
    if not targets:
        targets = np.arange(num_nodes)
    if not sources:
        sources = range(num_nodes)
    for source in sources:
        _, pred_map = shortest_distance(graph, source=source, target=targets, weights=weights, pred_map=True)
        flows_on_shortest_e += sum_flows_from_tree(
            source=source,
            targets=targets,
            pred_map_arr=np.array(pred_map.a),
            traffic_mat=traffic_mat,
            edge_to_ind=edge_to_ind,
        )

    return flows_on_shortest_e


def get_graphtool_graph(nx_graph: nx.Graph) -> gt.Graph:
    """Creates `gt_graph: graph_tool.Graph` from `nx_graph: nx.Graph`.
    Nodes in `gt_graph` are labeled by their indices in `nx_graph.edges()` instead of their labels
    (possibly of `str` type) in `nx_graph`"""

    def edge_dict_to_arr(edge_dict: dict, edge_to_ind: dict) -> np.ndarray:
        arr = np.zeros(len(edge_dict))
        for edge, value in edge_dict.items():
            arr[edge_to_ind[edge]] = value
        return arr

    nx_edges = nx_graph.edges()
    nx_edge_to_ind = dict(zip(nx_edges, list(range(len(nx_graph.edges())))))

    nx_bandwidths = edge_dict_to_arr(nx.get_edge_attributes(nx_graph, "bandwidth"), nx_edge_to_ind)
    nx_costs = edge_dict_to_arr(nx.get_edge_attributes(nx_graph, "cost"), nx_edge_to_ind)

    nx_nodes = list(nx_graph.nodes())
    edge_list = []
    for i, e in enumerate(nx_graph.edges()):
        edge_list.append((*[nx_nodes.index(v) for v in e], nx_bandwidths[i], nx_costs[i]))

    gt_graph = gt.Graph(edge_list, eprops=[("bandwidths", "double"), ("costs", "double")])

    return gt_graph


def maybe_create_and_get_ep(
    graph: gt.Graph,
    values: np.ndarray,
    edge_property_name: str = "weights",
) -> gt.EdgePropertyMap:
    """Creates (if not exists) an edge property, fills with `values` and returns"""

    if edge_property_name not in graph.edge_properties:
        ep = graph.new_edge_property("double")
        graph.ep[edge_property_name] = ep

    graph.ep[edge_property_name].a = values
    return graph.ep[edge_property_name]
