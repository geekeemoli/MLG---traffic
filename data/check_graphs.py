import pickle
import networkx as nx
import json
import math
import numpy as np
from load_graph import load_graph_nx as load_graph

def add_curvature(node_data):
    """
    Calculate curvature for a node based on its geometry. The curvature is defined as the ratio of the path length
    along the polyline to the straight-line distance between the endpoints, normalized so that straight lines have curvature 0.
    The curvature is clamped to [0, 5] and then log-scaled.

    Args:
        node_data (dict): Node data containing a 'geometry' key with a shapely LineString.
    Returns:
        dict: Node data with an added 'curvature' key.
    """
    default_return = { **node_data, "curvature": 0 }

    geom = node_data.get("geometry")
    if geom is None or geom.is_empty:
        return default_return

    coords = list(geom.coords)
    if len(coords) < 2:
        return default_return

    # path length along the polyline
    path_length = geom.length

    # straight-line distance between first and last point
    x0, y0 = coords[0]
    x1, y1 = coords[-1]
    chord_length = math.hypot(x1 - x0, y1 - y0)

    if chord_length == 0:
        return default_return

    curvature = path_length / chord_length
    curvature = curvature - 1  # normalize so that straight lines have curvature 0
    curvature = min(max(curvature, 0), 5)  # clamp to [0, 5]
    curvature = np.log1p(curvature) # apply log scaling

    return { **node_data, "curvature": curvature }


def main(path):
    graph = load_graph(path)

    # print(f"There are {len(graph.nodes)} nodes and {len(graph.edges)} edges in the graph.")
    
    all_node_attributes = set()
    for node_id in graph.nodes:
        all_node_attributes.update(graph.nodes[node_id].keys())
    # print("All node attribute names:", all_node_attributes)

    # count how many nodes have non-zero correlation
    non_zero_correlation = 0
    for node_id in graph.nodes:
        if graph.nodes[node_id].get("correlation", 0) != 0:
            non_zero_correlation += 1
    # print(f"Number of nodes with non-zero correlation: {non_zero_correlation}/{len(graph.nodes)}")
    non_zero_curvature = 0
    for node_id in graph.nodes:
        if graph.nodes[node_id].get("curvature", 0) != 0:
            non_zero_curvature += 1
    # print(f"Number of nodes with non-zero curvature: {non_zero_curvature}/{len(graph.nodes)}")
    path_file_name = path.split("/")[-1]
    print(f"| {path_file_name} | {len(graph.nodes)} | {len(graph.edges)} | {non_zero_correlation} |")

    node_ids = list(graph.nodes)
    step = max(1, len(node_ids) // 2)
    # for i in range(0, len(node_ids), step):
    #     node_id = node_ids[i]
        
    #     if "geometry" in graph.nodes[node_id]:
    #         print(f"Node ID: {node_id}\nData:", json.dumps({**add_curvature(graph.nodes[node_id]), "geometry": None}, indent=2))
    #     else:
    #         print(f"Node ID: {node_id}\nData:")
    #         for key in sorted(list(graph.nodes[node_id].keys())):
    #             print(f"  {key}: {graph.nodes[node_id][key]}")
    #     print("=" * 40)

    all_highway_attr_values = {}
    for i in range(0, len(node_ids)):
        node_id = node_ids[i]
        highway_attr = graph.nodes[node_id].get("highway")
        if highway_attr is not None:
            if isinstance(highway_attr, list):
                for val in highway_attr:
                    all_highway_attr_values[val] = all_highway_attr_values.get(val, 0) + 1
            else:
                all_highway_attr_values[highway_attr] = all_highway_attr_values.get(highway_attr, 0) + 1
    # print(f"All unique highway attribute values in this graph: {list(all_highway_attr_values)}")
    return all_highway_attr_values

# path = "./graphs/Graz_Austria_road_graph_with_popdensity.gpickle"
# path = "processed_graphs/Graz_Austria_graph.gpickle"
# paths = [
#     './graphs/Graz_Austria_road_graph_with_popdensity.gpickle',
#     './graphs/東京23区_Japan_road_graph_with_popdensity.gpickle',
#     './graphs/Melbourne_Australia_road_graph_with_popdensity.gpickle',
# ]
# paths should be all files in final_graphs directory ending with .gpickle
import os
paths = [os.path.join("final_graphs", f) for f in os.listdir("final_graphs") if f.endswith(".pkl.gz")]


highway_attr_values = {}
for path in paths:
    # print(f"Analyzing graph at: {path}")
    curr_highway_attr_values = main(path)
    for key, value in curr_highway_attr_values.items():
        highway_attr_values[key] = highway_attr_values.get(key, 0) + value
    # print("#" * 80)

print("=" * 80)
print(f"There are {len(highway_attr_values)} unique highway attribute values across all graphs.")
print(f"All unique highway attribute values across all graphs:")
for key, value in sorted(highway_attr_values.items(), key=lambda item: item[1], reverse=True):
    print(f"  {key}: {value}")
print("=" * 80)