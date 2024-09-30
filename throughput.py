from typing import List, Tuple, Optional
from dataclasses import dataclass

import cvxpy as cp
import networkx as nx
import numpy as np

FLOAT = np.float64

from utils import calculate_laplacian_from_weights_matrix, get_incidence_matrix, get_var_value, get_weights


@dataclass(frozen=True)
class Solution:
    problem: cp.Problem
    flow: np.ndarray
    add_bandwidth: Optional[np.ndarray] = None
    gamma: Optional[float] = None

    def to_vector(self) -> np.ndarray:
        sol_parts = [self.flow.T.flatten()]
        if self.add_bandwidth is not None:
            sol_parts.append(self.add_bandwidth)
        if self.gamma is not None:
            sol_parts.append(self.gamma)
        return np.hstack(sol_parts)

def solve_throughput(graph: nx.Graph, traffic_mat: np.ndarray, **solver_kwargs) -> Solution:
    graph = nx.DiGraph(graph)
    traffic_lapl = calculate_laplacian_from_weights_matrix(traffic_mat, "out")
    incidence_mat = get_incidence_matrix(graph)
    bandwidth = get_weights(graph, "bandwidth")

    flow = cp.Variable((len(graph.edges), traffic_mat.shape[0]))
    gamma = cp.Variable()
    prob = cp.Problem(
        cp.Maximize(gamma),
        [cp.sum(flow, axis=1) <= bandwidth, incidence_mat @ flow == -gamma * traffic_lapl.T, flow >= 0, gamma >= 0],
    )
    prob.solve(**solver_kwargs)

    if prob.status != "optimal":
        gamma = None

    return Solution(problem=prob, flow=get_var_value(flow), gamma=get_var_value(gamma))


def optimize_throughput(graph: nx.Graph, traffic_mat: np.ndarray, budget: float, **solver_kwargs) -> Solution:
    graph = nx.DiGraph(graph)
    traffic_lapl = calculate_laplacian_from_weights_matrix(traffic_mat, "out")
    incidence_mat = get_incidence_matrix(graph)
    bandwidth = get_weights(graph, "bandwidth")

    flow = cp.Variable((len(graph.edges), traffic_mat.shape[0]))
    add_bandwidth = cp.Variable(len(graph.edges))
    gamma = cp.Variable()
    prob = cp.Problem(
        cp.Maximize(gamma),
        [
            cp.sum(flow, axis=1) <= bandwidth + add_bandwidth,
            incidence_mat @ flow == -gamma * traffic_lapl.T,
            cp.sum(add_bandwidth) <= budget,
            flow >= 0,
            gamma >= 0,
            add_bandwidth >= 0,
        ],
    )
    prob.solve(**solver_kwargs)

    if prob.status != "optimal":
        gamma = None

    return Solution(
        problem=prob, flow=get_var_value(flow), add_bandwidth=get_var_value(add_bandwidth), gamma=get_var_value(gamma)
    )


def optimize_robust_throughput(
    graph: nx.DiGraph,
    traffic_mat: np.ndarray,
    budget: float,
    proportion_edge_perturbed: float = 1.0,
    **solver_kwargs,
) -> Solution:
    traffic_lapl = calculate_laplacian_from_weights_matrix(traffic_mat, "out")
    incidence_mat = get_incidence_matrix(graph)
    bandwidth = get_weights(graph, "bandwidth")

    num_edges = len(graph.edges)
    num_nodes = len(graph.nodes)
    flow = [cp.Variable((num_edges, num_nodes)) for _ in range(num_edges)]
    add_bandwidth = cp.Variable(num_edges)
    gamma = cp.Variable()
    # t = cp.Variable()

    kirchhoff_constr = [(incidence_mat @ f == -gamma * traffic_lapl.T) for f in flow]
    bandwidth_constr = []
    for e in range(num_edges):
        rhs_mul = np.ones_like(bandwidth)
        rhs_mul[e] = 1 - proportion_edge_perturbed
        bandwidth_constr += [cp.sum(flow[e], axis=1) <= cp.multiply(bandwidth + add_bandwidth, rhs_mul)]

    prob = cp.Problem(
        cp.Maximize(gamma),
        kirchhoff_constr
        + bandwidth_constr
        #+ [gamma >= t * np.ones_like(gamma)]
        + [f >= 0 for f in flow]
        + [gamma >= 0, add_bandwidth >= 0, cp.sum(add_bandwidth) <= budget],
    )
    prob.solve(**solver_kwargs)

    if prob.status != "optimal":
        gamma = None

    return Solution(
        problem=prob,
        flow=[get_var_value(f) for f in flow],
        add_bandwidth=get_var_value(add_bandwidth),
        gamma=get_var_value(gamma),
    )
