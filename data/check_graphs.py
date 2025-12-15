import pickle
import networkx as nx
import json
import math
import numpy as np

def load_graph(path):
    with open(path, "rb") as f:
        graph = pickle.load(f)
    return graph

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

    print(f"There are {len(graph.nodes)} nodes and {len(graph.edges)} edges in the graph.")

    print("Sample node data:")
    # print the node data for 10 nodes spaced evenly throughout the graph
    node_ids = list(graph.nodes)
    step = max(1, len(node_ids) // 10)
    for i in range(0, len(node_ids), step):
        node_id = node_ids[i]
        print(f"Node ID: {node_id}\nData:", json.dumps({**add_curvature(graph.nodes[node_id]), "geometry": None}, indent=2))
        # print("Node ID:", node_id, "Attribute names:", list(graph.nodes[node_id].keys()))
        print("=" * 40)

# path = "./graphs/Graz_Austria_road_graph_with_popdensity.gpickle"
# path = "processed_graphs/Graz_Austria_graph.gpickle"
paths = [
    './graphs/Graz_Austria_road_graph_with_popdensity.gpickle'
    './graphs/Tokyo_Japan_road_graph_with_popdensity.gpickle'
    './graphs/Melbourne_Australia_road_graph_with_popdensity.gpickle'
]
for path in paths:
    print(f"Analyzing graph at: {path}")
    main(path)
    print("#" * 80)
    print("\n\n")