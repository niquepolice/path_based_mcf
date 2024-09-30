import graph_tool as gt
import networkx as nx
import numpy as np
from graph_tool.topology import shortest_distance
from utils import get_index, get_path_edges, plot_convergence, shortest_paths_lengths
from tqdm import tqdm
from typing import Optional

from utils import flows_on_shortest_gt, get_graphtool_graph
from utils import calculate_laplacian_from_weights_matrix, get_incidence_matrix, get_weights


def vanilla_switching(
    graph: nx.Graph,
    traffic_mat: np.ndarray,
    max_iters: int,
    h_f: float,
    h_g: float,
    epsilon: float,
    throughput: float,
    verbose: bool = False,
):
    A = get_incidence_matrix(graph)
    gt_graph = get_graphtool_graph(graph)
    bandwidths = gt_graph.edge_properties["bandwidths"]
    num_nodes, num_edges = gt_graph.num_vertices(), gt_graph.num_edges()

    z = np.zeros((num_nodes, num_nodes))

    weights = gt_graph.new_edge_property("double")
    weights.a = np.ones(num_edges)

    flows = gt_graph.new_edge_property("double")
    flows.a = np.zeros(num_edges)

    # Словарь для хранения индексов рёбер
    flows_ei = np.zeros((num_edges, num_nodes))

    kirch = []
    lam_delta = []
    traffic_lapl = calculate_laplacian_from_weights_matrix(traffic_mat, "out")

    I = 0
    second_count = 0

    # rng = tqdm(range(max_iters)) if verbose else range(max_iters)
    one_element_traffic_mat = np.zeros((num_nodes, num_nodes))
    for k in tqdm(range(max_iters)):
        # матрица длин кратчайших путей
        # элемент (i, j) содержит длину кратчайшего пути от узла i до узла j.
        dist_matrix = shortest_paths_lengths(gt_graph, weights)
        #     dist_matrix,kj
        second_constr = 1 - (traffic_mat * z).sum()
        # первое и второе ограничения проверяем вместе
        first = np.all(z - dist_matrix <= epsilon)
        second = second_constr <= epsilon

        if first and second:
            # print('Продуктивный шаг')
            I += 1
            # Шаг по weights
            weights.a = np.maximum(0, weights.a - h_f * bandwidths.a)
        else:
            # print('Непродуктивный шаг')
            # определение максимально нарушенного ограничения
            all_ineqs_values = np.append((z - dist_matrix).flatten(), second_constr)

            max_viol_num = np.argmax(all_ineqs_values)
            if max_viol_num == all_ineqs_values.size - 1:
                # print('Второе ограничение нарушено')
                z = z + h_g * traffic_mat
                second_count += 1  # h_g
            else:
                # print('Первое ограничение нарушено')
                # Шаг по weights
                source_idx, target_idx = get_index(dist_matrix.shape, max_viol_num)

                one_element_traffic_mat[source_idx, target_idx] = 1  # make sure to erase after use
                flows_on_path = flows_on_shortest_gt(
                    gt_graph,
                    one_element_traffic_mat,
                    weights,
                    sources=np.array([source_idx]),
                    targets=np.array([target_idx]),
                )
                # print(f"{flows_on_path=}")
                one_element_traffic_mat[source_idx, target_idx] = 0  # erasing
                flows_ei[:, source_idx] += flows_on_path
                flows.a += flows_on_path
                weights.a += h_g * flows_on_path

                # Шаг по ограничению
                z[source_idx, target_idx] = z[source_idx, target_idx] - h_g

        if I > 0:
            lambda_averaged = second_count * h_g / (I * h_f)
            flows_ei_averaged = flows_ei * h_g / (I * h_f)
            flows_averaged = flows_ei_averaged.sum(axis=1)
            lam_delta.append(abs(lambda_averaged - throughput))
            if verbose:
                kirch.append(np.sum(np.abs(A @ flows_ei_averaged + lambda_averaged * (traffic_lapl).T)))
                if not k % (max_iters // 5):
                    print(lambda_averaged)
                    plot_convergence(flows_averaged, bandwidths, kirch, k, None, throughput, lam_delta)
    return lambda_averaged, flows_averaged, lam_delta


def no_potentials_switching(
    graph: nx.Graph,
    traffic_mat: np.ndarray,
    max_iters: int,
    h_f: float,
    h_g: float,
    epsilon: float,
    throughput: float,
    verbose: bool = False,
):
    A = get_incidence_matrix(graph)
    gt_graph = get_graphtool_graph(graph)
    bandwidths = gt_graph.edge_properties["bandwidths"]
    num_nodes, num_edges = gt_graph.num_vertices(), gt_graph.num_edges()

    weights = gt_graph.new_edge_property("double")
    weights.a = np.ones(num_edges)

    flows_sum = np.zeros(num_edges)

    lam_delta = []
    traffic_lapl = calculate_laplacian_from_weights_matrix(traffic_mat, "out")

    I = 0
    second_count = 0

    # rng = tqdm(range(max_iters)) if verbose else range(max_iters)
    one_element_traffic_mat = np.zeros((num_nodes, num_nodes))
    for k in tqdm(range(max_iters)):
        flows_aon = flows_on_shortest_gt(gt_graph, traffic_mat, weights)
        constr_lhs = 1 - flows_aon @ weights.a

        # productive step
        if constr_lhs <= epsilon:
            I += 1
            weights.a = np.maximum(0, weights.a - h_f * bandwidths.a)
        else:
            weights.a += h_g * flows_aon
            flows_sum += flows_aon

        if I > 0:
            flows_averaged = flows_sum * h_g / (I * h_f)
            lambda_averaged = (k + 1 - I)  * h_g / (I * h_f)
            lam_delta.append(abs(lambda_averaged - throughput))
            if verbose:
                if not k % (max_iters // 5):
                    print(lambda_averaged)
                    kirch = [0]
                    plot_convergence(flows_averaged, bandwidths, kirch, k, None, throughput, lam_delta)
    return lambda_averaged, flows_averaged, lam_delta


def regularized_gd(
    graph: nx.Graph,
    traffic_mat: np.ndarray,
    max_iters: int,
    h: float,
    mu: float,
    throughput: float,
    verbose: bool = False,
):
    A = get_incidence_matrix(graph)
    gt_graph = get_graphtool_graph(graph)
    num_nodes, num_edges = gt_graph.num_vertices(), gt_graph.num_edges()
    bandwidths = gt_graph.edge_properties["bandwidths"]

    weights = gt_graph.new_edge_property("double")
    weights.a = np.zeros(num_edges)

    kirch = []
    lam_delta = []
    traffic_lapl = calculate_laplacian_from_weights_matrix(traffic_mat, "out")

    flows_sum = np.zeros(num_edges)

    lambda_sum = 0

    for k in tqdm(range(max_iters)):
        flows_aon = flows_on_shortest_gt(gt_graph, traffic_mat, weights)
        total_cost = (flows_aon @ weights.a).sum()
        lambda_y = max(0, (1 - total_cost) / mu)
        weights.a = np.maximum(0, weights.a - h * (bandwidths.a - (lambda_y > 0) * lambda_y * flows_aon))

        lambda_sum += lambda_y

        flows_sum += flows_aon * lambda_y

        lam_delta.append(abs((lambda_sum / (k + 1)) - throughput))

        lambda_averaged = lambda_sum / (k + 1)
        flows_averaged = flows_sum / (k + 1)
        if verbose:
            kirch.append(
                np.linalg.norm(A @ (flows_sum / (k + 1)) + (lambda_sum / (k + 1)) * (traffic_lapl).T.sum(axis=1))
            )
            if not k % (max_iters // 5):
                print(lambda_averaged)
                plot_convergence(flows_averaged, bandwidths, kirch, k, None, throughput, lam_delta)

    return lambda_averaged, flows_averaged, lam_delta


def get_Lambda(graph: nx.Graph, traffic_mat: np.ndarray, budget: float = 0) -> float:
    A_weighted = nx.incidence_matrix(
        graph, oriented=True, weight="bandwidth"
    ).todense()  # A * bandwidths.a[np.newaxis, :]
    A_in = A_weighted * (A_weighted > 0)
    A_out = -A_weighted * (A_weighted < 0)

    inflow, outflow = traffic_mat.sum(axis=0), traffic_mat.sum(axis=1)

    inflow_caps = A_in.sum(axis=1) + budget
    outflow_caps = A_out.sum(axis=1) + budget

    mask_in = inflow > 0
    mask_out = outflow > 0
    Lambda = min((inflow_caps[mask_in] / inflow[mask_in]).min(), (outflow_caps[mask_out] / outflow[mask_out]).min())
    return Lambda


def bounded_ugd(
    graph: nx.Graph,
    traffic_mat: np.ndarray,
    max_iters: int,
    h0: float,
    eps: Optional[float],  # set to None to disable backtracking
    throughput: float,
    verbose: bool = False,
):
    Lambda = get_Lambda(graph, traffic_mat)
    print(Lambda)

    use_backtracking = eps is not None

    gt_graph = get_graphtool_graph(graph)
    num_nodes, num_edges = gt_graph.num_vertices(), gt_graph.num_edges()
    bandwidths = gt_graph.edge_properties["bandwidths"]
    A = get_incidence_matrix(graph)
    traffic_lapl = calculate_laplacian_from_weights_matrix(traffic_mat, "out")

    weights = gt_graph.new_edge_property("double")
    weights.a = np.zeros(num_edges)

    kirch = []
    lam_delta = []

    flows_averaged = np.zeros(num_edges)

    lambda_averaged = 0

    def dual_func(weights, flows_aon):
        total_cost = (flows_aon @ weights.a).sum()
        return weights.a @ bandwidths.a + max(0, Lambda * (1 - total_cost))

    h = h0
    h_sum = 0

    flows_aon_new = dual_func_new = total_cost_new = lambda_y_new = None

    h_trace = []

    for k in tqdm(range(max_iters)):
        if verbose:
            kirch.append(np.linalg.norm(A @ flows_averaged + lambda_averaged * traffic_lapl.T.sum(axis=1)))
            if k > 0 and not k % (max_iters // 5) or len(lam_delta) >= max_iters:
                delta = 0
                print(lambda_averaged)
                plot_convergence(flows_averaged, bandwidths, kirch, k, delta, throughput, lam_delta)


        if len(lam_delta) >= max_iters:
            break

        if flows_aon_new is None:
            flows_aon = flows_on_shortest_gt(gt_graph, traffic_mat, weights)
            dual_func_prev = dual_func(weights, flows_aon)
            total_cost = (flows_aon @ weights.a).sum()
            lambda_y = Lambda * ((1 - total_cost) > 0)
        else:
            flows_aon = flows_aon_new
            dual_func_prev = dual_func_new
            lambda_y = lambda_y_new

        weights_prev = weights.a.copy()
        grad_y = bandwidths.a - lambda_y * flows_aon

        # import matplotlib.pyplot as plt
        # if not k % 100:
        #     hs = np.logspace(-2, 0, 100)
        #     dual_funcs = []
        #     for h_ in hs:
        #         weights.a = np.maximum(0, weights_prev - h_ * grad_y)
        #         flows_aon_new = flows_on_shortest_gt(gt_graph, traffic_mat, weights)

        #         total_cost_new = (flows_aon_new @ weights.a).sum()
        #         lambda_y_new = Lambda * ((1 - total_cost_new) > 0)
        #         inner_prod = (weights.a - weights_prev) @ grad_y
        #         dual_func_new = dual_func(weights, flows_aon_new)
        #         dual_funcs.append(dual_func_new)

        #     plt.plot(hs, dual_funcs)
        #     plt.xscale("log")

        # break

        if use_backtracking:
            h *= 2

        while True:
            weights.a = np.maximum(0, weights_prev - h * grad_y)
            flows_aon_new = flows_on_shortest_gt(gt_graph, traffic_mat, weights)

            lam_delta.append(abs(lambda_averaged - throughput))  # add here to correctly count shortest paths calls

            total_cost_new = (flows_aon_new @ weights.a).sum()
            lambda_y_new = Lambda * ((1 - total_cost_new) > 0)
            if not use_backtracking:
                break

            inner_prod = (weights.a - weights_prev) @ grad_y
            dual_func_new = dual_func(weights, flows_aon_new)
            lhs = dual_func_new - dual_func_prev
            rhs = inner_prod + ((weights.a - weights_prev) ** 2).sum() / (2 * h)
            # print(f"{lhs - inner_prod =}")
            # print(f"{lhs - (rhs + eps / 2) =}")
            if lhs <= rhs + eps / 2:
            #    print()
                break
            h /= 2

        h_trace.append(h)
        lambda_averaged = (lambda_y * h + lambda_averaged * h_sum) / (h + h_sum)
        flows_averaged = (lambda_y * flows_aon * h + flows_averaged * h_sum) / (h + h_sum)
        h_sum += h

    import matplotlib.pyplot as plt
    plt.plot(h_trace)
    plt.yscale("log")

    return lambda_averaged, flows_averaged, lam_delta

def budget_switching(
    graph: nx.Graph,
    traffic_mat: np.ndarray,
    max_iters: int,
    h_f: float,
    h_g: float,
    epsilon: float,
    budget: float,
    throughput: float,
    verbose: bool = False,
    start_point = None
):
    A = get_incidence_matrix(graph)
    gt_graph = get_graphtool_graph(graph)
    bandwidths = gt_graph.edge_properties["bandwidths"]
    num_nodes, num_edges = gt_graph.num_vertices(), gt_graph.num_edges()

    if start_point is None:
        z_ij = np.zeros((num_nodes, num_nodes))
        y_e = np.zeros((num_edges))
        t = 0
    else:
        z_ij, y_e, t = start_point[0].copy(), start_point[1].copy(), start_point[2]


    l_ij = np.zeros((num_nodes, num_nodes))

    # weights = [gt_graph.new_edge_property("double") for _ in range(num_edges)]
    weights = gt_graph.new_edge_property("double")
    # weights.a = np.ones(num_edges)

    flows = gt_graph.new_edge_property("double")
    flows.a = np.zeros(num_edges)

    flows_ei = np.zeros((num_edges, num_nodes))

    lambda_sum = 0
    add_bw_sum = np.zeros(num_edges)

    kirch = []
    lam_delta = []
    traffic_lapl = calculate_laplacian_from_weights_matrix(traffic_mat, "out")

    I = 0
    

    # rng = tqdm(range(max_iters)) if verbose else range(max_iters)
    one_element_traffic_mat = np.zeros((num_nodes, num_nodes))
    for k in tqdm(range(max_iters)):
        weights.a = y_e
        l_ij = shortest_paths_lengths(gt_graph, weights)  # dist_matrix
        
        dz_constr = 1 - (traffic_mat * z_ij).sum() # <= epsilon
        lz_constr = z_ij - l_ij # <= epsilon
        yt_constr = y_e - t

        three_constrs = [dz_constr, lz_constr.max(), yt_constr.max()]
        # print(three_constrs)
        if max(three_constrs) > epsilon:
            max_constr = ["dz", "lz", "yt"][np.argmax(three_constrs)] 
            # print(k, three_constrs, max_constr)
            if max_constr == "dz":
                lambda_sum += 1
                z_ij = z_ij + h_g * traffic_mat
                
            if max_constr == "lz":
                max_index = np.argmax(lz_constr)
                ij_max = np.unravel_index(max_index, z_ij.shape)
                source_idx, target_idx = ij_max 
                z_ij[ij_max] = max(0, z_ij[ij_max] - h_g)

                one_element_traffic_mat[source_idx, target_idx] = 1  # make sure to erase after use
                weights.a = y_e
                flows_on_path = flows_on_shortest_gt(
                    gt_graph,
                    one_element_traffic_mat,
                    weights,
                    sources=np.array([source_idx]),
                    targets=np.array([target_idx]),
                )
                one_element_traffic_mat[source_idx, target_idx] = 0  # erasing

                flows_ei[:, source_idx] += flows_on_path
                flows.a += flows_on_path
                y_e += h_g * flows_on_path

            if max_constr == "yt":
                # yt_constr = y_qe.sum(axis=0) - alpha * np.diag(y_qe) - t 
                t += h_g
                e_max = np.argmax(yt_constr)
                add_bw_sum[e_max] += 1
                y_e[e_max] -= h_g
                y_e = np.maximum(0, y_e)

        else:  # productive step
            # print("prod")
            I += 1
            t = max(0, t - budget * h_f)
            grad_y = bandwidths.a
            
            y_e = np.maximum(0, y_e - h_f * grad_y)

        if I > 0:
            lambda_averaged = lambda_sum * h_g / (I * h_f)
            lam_delta.append(abs(lambda_averaged - throughput))
            flows_ei_averaged = flows_ei * h_g / (I * h_f)
            flows_averaged = flows.a / I * h_g / h_f 
            add_bw_averaged = add_bw_sum * h_g / (I * h_f)

            if verbose:
                kirch.append(np.sum(np.abs(A @ flows_ei_averaged + lambda_averaged * (traffic_lapl).T)))
                if not k % (max_iters // 5) or k == max_iters - 1:
                    print(lambda_averaged)
                    # delta = flows.a / I * h_g / h_f - weights.a
                    plot_convergence(flows_averaged, bandwidths, kirch, k, add_bw_averaged, throughput, lam_delta)

    return lambda_averaged, flows_averaged, add_bw_averaged, lam_delta, (z_ij, y_e, t)



def robust_switching(
    graph: nx.Graph,
    traffic_mat: np.ndarray,
    max_iters: int,
    h_f: float,
    h_g: float,
    epsilon: float,
    budget: float,
    proportion_edge_perturbed: float,
    throughput: float,
    verbose: bool = False,
    start_point = None
):
    alpha = proportion_edge_perturbed
    A = get_incidence_matrix(graph)
    gt_graph = get_graphtool_graph(graph)
    bandwidths = gt_graph.edge_properties["bandwidths"]
    num_nodes, num_edges = gt_graph.num_vertices(), gt_graph.num_edges()

    if start_point is None:
        z_qij = np.zeros((num_edges, num_nodes, num_nodes))
        y_qe = np.zeros((num_edges, num_edges))
        t = 0
    else:
        z_qij, y_qe, t = start_point[0].copy(), start_point[1].copy(), start_point[2]


    l_qij = np.zeros((num_edges, num_nodes, num_nodes))

    # weights = [gt_graph.new_edge_property("double") for _ in range(num_edges)]
    weights = gt_graph.new_edge_property("double")
    # weights.a = np.ones(num_edges)

    flows = gt_graph.new_edge_property("double")
    flows.a = np.zeros(num_edges)

    flows_ei = np.zeros((num_edges, num_nodes))

    lambda_sum = 0
    add_bw_sum = np.zeros(num_edges)

    kirch = []
    lam_delta = []
    traffic_lapl = calculate_laplacian_from_weights_matrix(traffic_mat, "out")

    I = 0
    

    # rng = tqdm(range(max_iters)) if verbose else range(max_iters)
    one_element_traffic_mat = np.zeros((num_nodes, num_nodes))
    for k in tqdm(range(max_iters)):
        for q in range(num_edges):
            weights.a = y_qe[q]
            l_qij[q] = shortest_paths_lengths(gt_graph, weights)  # dist_matrix
        
        dz_constr = 1 - (traffic_mat * z_qij.sum(axis=0)).sum() # <= epsilon
        lz_constr = z_qij - l_qij # <= epsilon
        yt_constr = y_qe.sum(axis=0) - alpha * np.diag(y_qe) - t # <= epsilon

        three_constrs = [dz_constr, lz_constr.max(), yt_constr.max()]
        # print(three_constrs)
        if max(three_constrs) > epsilon:
            max_constr = ["dz", "lz", "yt"][np.argmax(three_constrs)] 
            # print(k, three_constrs, max_constr)
            if max_constr == "dz":
                lambda_sum += 1
                z_qij = z_qij + h_g * traffic_mat[np.newaxis, ...] 
                
            if max_constr == "lz":
                max_index = np.argmax(lz_constr)
                qij_max = np.unravel_index(max_index, z_qij.shape)
                z_qij[qij_max] = max(0, z_qij[qij_max] - h_g)
             
                

                source_idx, target_idx = qij_max[1:]

                one_element_traffic_mat[source_idx, target_idx] = 1  # make sure to erase after use
                weights.a = y_qe[qij_max[0]]
                flows_on_path = flows_on_shortest_gt(
                    gt_graph,
                    one_element_traffic_mat,
                    weights,
                    sources=np.array([source_idx]),
                    targets=np.array([target_idx]),
                )
                one_element_traffic_mat[source_idx, target_idx] = 0  # erasing

                flows_ei[:, source_idx] += flows_on_path
                flows.a += flows_on_path
                y_qe[qij_max[0], :] += h_g * flows_on_path

            if max_constr == "yt":
                # yt_constr = y_qe.sum(axis=0) - alpha * np.diag(y_qe) - t 
                t += h_g
                e_max = np.argmax(yt_constr)
                add_bw_sum[e_max] += 1
                y_qe[:, e_max] -= h_g
                y_qe[e_max, e_max] += alpha * h_g
                y_qe = np.maximum(0, y_qe)

        else:  # productive step
            # print("prod")
            I += 1
            t = max(0, t - budget * h_f)
            grad = -(alpha * np.diag(bandwidths.a) - bandwidths.a[np.newaxis, :])
            
            y_qe = np.maximum(0, y_qe - h_f * grad)

        if I > 0:
            lambda_averaged = lambda_sum * h_g / (I * h_f)
            lam_delta.append(abs(lambda_averaged - throughput))
            flows_ei_averaged = flows_ei * h_g / (I * h_f)
            flows_averaged = flows.a / I * h_g / h_f 
            add_bw_averaged = add_bw_sum * h_g / (I * h_f)

            if verbose:
                kirch.append(np.sum(np.abs(A @ flows_ei_averaged + lambda_averaged * (traffic_lapl).T)))
                if not k % (max_iters // 5) or k == max_iters - 1:
                    print(lambda_averaged)
                    # delta = flows.a / I * h_g / h_f - weights.a
                    plot_convergence(flows_averaged, bandwidths, kirch, k, add_bw_averaged, throughput, lam_delta)

    return lambda_averaged, flows_averaged, add_bw_averaged, lam_delta, (z_qij, y_qe, t)


def bounded_robust_gd(
    graph: nx.Graph,
    traffic_mat: np.ndarray,
    max_iters: int,
    h0: float,
    eps: Optional[float],  # set to None to disable backtracking
    budget: float,
    proportion_edge_perturbed: float,
    throughput: float,
    verbose: bool = False,
    y_qe0 = None,
):
    Lambda = get_Lambda(graph, traffic_mat, budget)
    print(Lambda)
    alpha = proportion_edge_perturbed

    use_backtracking = eps is not None

    gt_graph = get_graphtool_graph(graph)
    num_nodes, num_edges = gt_graph.num_vertices(), gt_graph.num_edges()
    bandwidths = gt_graph.edge_properties["bandwidths"]
    A = get_incidence_matrix(graph)
    traffic_lapl = calculate_laplacian_from_weights_matrix(traffic_mat, "out")

    weights = gt_graph.new_edge_property("double")
    weights.a = np.zeros(num_edges)

    kirch = []
    lam_delta = []

    lambda_averaged = 0

    def get_flows_aon_qe(y_qe):
        if alpha == 0:
            weights.a = y_qe[0]
            f = flows_on_shortest_gt(gt_graph, traffic_mat, weights)
            for q in range(num_edges):
                weights.a = y_qe[q]
                flows_aon_qe[q] = f
        else:
            for q in range(num_edges):
                weights.a = y_qe[q]
                flows_aon_qe[q] = flows_on_shortest_gt(gt_graph, traffic_mat, weights)

        return flows_aon_qe

    def get_dual_f_lambda(y_qe, flows_aon_qe):
        total_cost = (flows_aon_qe.sum(axis=0) @ weights.a).sum()

        lambda_y = Lambda * ((1 - total_cost) > 0)

        dual_f = (lambda_y * (1 - total_cost) 
                  + budget * (y_qe.sum(axis=0) - alpha * y_qe.diagonal()).max()
                  + bandwidths.a @ (y_qe.sum(axis=0) - alpha * y_qe.diagonal())
                 )

        return dual_f, lambda_y

    h = h0
    h_sum = 0

    flows_averaged_qe = flows_aon_qe = y_qe = np.zeros((num_edges, num_edges))
    add_bw_averaged = np.zeros(num_edges)
    if y_qe0 is not None:
        y_qe = y_qe0.copy()

    # flows_aon_new = dual_func_new = total_cost_new = lambda_y_new = None

    h_trace = []

    flows_aon_qe = get_flows_aon_qe(y_qe)
    dual_f, lambda_y = get_dual_f_lambda(y_qe, flows_aon_qe)

    for k in tqdm(range(max_iters)):
        if verbose:
            kirch.append(np.linalg.norm(A @ flows_averaged_qe.mean(axis=0) + lambda_averaged * traffic_lapl.T.sum(axis=1)))
            if k > 0 and not k % (max_iters // 5) or len(lam_delta) >= max_iters or k == max_iters - 1:
                print(lambda_averaged)
                plot_convergence(flows_averaged_qe.mean(axis=0), bandwidths, kirch, k, add_bw_averaged, throughput, lam_delta)

        if len(lam_delta) >= max_iters:
            break

        if use_backtracking:
            h *= 2

        e_max = np.argmax(y_qe.sum(axis=0) - alpha * y_qe.diagonal())

        add_bw = np.zeros(num_edges)
        add_bw[e_max] = budget

        grad_qe = -lambda_y * flows_aon_qe
        grad_qe[:, e_max] +=  budget
        grad_qe[e_max, e_max] -= budget * alpha
        grad_qe += bandwidths.a[np.newaxis, :]
        grad_qe -= alpha * np.diag(bandwidths.a)
        # grad_qe = np.maximum(0, y_qe)
        
        while True: 
            y_qe_new = np.maximum(0, y_qe - h * grad_qe)
            flows_aon_qe_new = get_flows_aon_qe(y_qe_new)
            dual_f_new, lambda_y_new = get_dual_f_lambda(y_qe_new, flows_aon_qe_new)

            lam_delta.append(abs(lambda_averaged - throughput))  # add here to correctly count shortest paths calls

            rhs = dual_f + (grad_qe * (y_qe_new - y_qe)).sum() + ((y_qe_new - y_qe) ** 2).sum() / (2 * h)
            if not use_backtracking or dual_f_new <= rhs + eps / 2:
                break
            else:
                h /= 2

        # q = 0.001

        h_trace.append(h)
        # lambda_averaged = lambda_y * q + lambda_averaged * (1 - q)
        lambda_averaged = (lambda_y * h + lambda_averaged * h_sum) / (h + h_sum)
        flows_averaged_qe = (lambda_y * flows_aon_qe * h + flows_averaged_qe * h_sum) / (h + h_sum)
        #add_bw_averaged = (budget * np.eye(1, num_edges, e_max)[0] * h + add_bw_averaged * h_sum) / (h + h_sum)

        # add_bw_averaged = (q * add_bw + (1 - q) * add_bw_averaged) 
        add_bw_averaged = (add_bw * h + add_bw_averaged * h_sum) / (h + h_sum)
        h_sum += h

        y_qe, flows_aon_qe, dual_f, lambda_y = y_qe_new, flows_aon_qe_new, dual_f_new, lambda_y_new

    import matplotlib.pyplot as plt
    plt.plot(h_trace)
    plt.yscale("log")

    return lambda_averaged, flows_averaged_qe, add_bw_averaged, lam_delta, y_qe
