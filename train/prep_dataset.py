import pickle
import networkx as nx
import gzip
import os
from torch_geometric.utils import from_networkx
from tqdm import tqdm

def load_graph_nx(path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)

def load_graph_pyg(path):
    graph_nx = load_graph_nx(path)
    graph_pyg = from_networkx(graph_nx)
    return graph_pyg

def load_dataset(max_cities=None):
    dataset = {}
    base_path = "../data/final_graphs/"
    file_list = os.listdir(base_path)
    if max_cities is not None:
        file_list = file_list[:max_cities]

    skip_cities = ["Melbourne", "Japan"]
    file_list = [f for f in file_list if not any(skip_city in f for skip_city in skip_cities)]

    for filename in tqdm(file_list, desc="Loading graphs"):
        if filename.endswith(".pkl.gz"):
            city_name = filename.replace("_graph.pkl.gz", "")
            dataset[city_name] = load_graph_pyg(os.path.join(base_path, filename))

    return dataset

if __name__ == "__main__":
    print("Testing the load_dataset function...")
    dataset = load_dataset()
    for city, graph in dataset.items():
        print(f"City: {city}, Graph: {graph}")