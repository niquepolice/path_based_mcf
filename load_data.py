import json
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import xmltodict

FLOAT = np.float64

def read_graph_sndlib_xml(filename: Path) -> nx.Graph:
    with open(filename, "r") as file:
        graph_dct = xmltodict.parse(file.read())["network"]["networkStructure"]

    graph = nx.DiGraph()

    for node in graph_dct["nodes"]["node"]:
        graph.add_node(node["@id"], x=FLOAT(node["coordinates"]["x"]), y=FLOAT(node["coordinates"]["y"]))

    for edge in graph_dct["links"]["link"]:
        cost = FLOAT(edge.get("routingCost", 1.0))
        if "preInstalledModule" in edge:
            bandwidth = FLOAT(edge["preInstalledModule"]["capacity"])
        elif "additionalModules" in edge and "addModule" in edge["additionalModules"]:
            module = edge["additionalModules"]["addModule"]
            if isinstance(module, list):
                module = module[0]
            bandwidth = FLOAT(module["capacity"])
        else:
            bandwidth = FLOAT(1.0)
        graph.add_edge(edge["source"], edge["target"], cost=cost, bandwidth=bandwidth)
        graph.add_edge(edge["target"], edge["source"], cost=cost, bandwidth=bandwidth)

    return graph


def read_traffic_mat_sndlib_xml(filename) -> np.ndarray:
    with open(filename, "r") as file:
        xml_dct = xmltodict.parse(file.read())["network"]

    node_label_to_num = {node["@id"]: i for i, node in enumerate(xml_dct["networkStructure"]["nodes"]["node"])}
    traffic_mat = np.zeros((len(node_label_to_num), len(node_label_to_num)), dtype=FLOAT)
    for demand in xml_dct["demands"]["demand"]:
        source = node_label_to_num[demand["source"]]
        target = node_label_to_num[demand["target"]]
        traffic_mat[source, target] = demand["demandValue"]
    return traffic_mat
    

def scale_graph_bandwidth_and_cost(graph: nx.Graph) -> nx.Graph:
    scaled_graph = graph.copy()
    max_bandwidth = max(nx.get_edge_attributes(graph, "bandwidth").values())
    max_cost = max(nx.get_edge_attributes(graph, "cost").values())
    for edge in graph.edges:
        scaled_graph.edges[edge]["bandwidth"] /= max_bandwidth
        scaled_graph.edges[edge]["cost"] /= max_cost
    return scaled_graph
