import pickle
import networkx as nx
import gzip
import os
from torch_geometric.utils import from_networkx

def load_graph_nx(path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)

def load_graph_pyg(path):
    graph_nx = load_graph_nx(path)
    graph_pyg = from_networkx(graph_nx)
    return graph_pyg

if __name__ == "__main__":
    print("Testing loading of PyG graph...")
    path = "./final_graphs/Graz_Austria_graph.pkl.gz"
    graph = load_graph_pyg(path)
    print(graph)