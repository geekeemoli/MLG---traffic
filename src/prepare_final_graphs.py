import osmnx as ox
import networkx as nx
import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Tuple
from collections import defaultdict
import os
import pickle
import csv
from io import StringIO
import math
import gzip

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_correlation_data(correlation_file: str = "correlations.csv") -> pd.DataFrame:
    """
    Load detector correlations from Dario's analysis.
    Uses only the correlation_ma column (moving average correlation).
    
    Args:
        correlation_file: Path to CSV with detector correlations
        
    Returns:
        DataFrame with detector_id and correlation_ma columns
    """
    df = pd.read_csv(correlation_file)
    
    df = df[['detector_id', 'correlation_ma']].copy()
    
    df = df.dropna(subset=['correlation_ma'])
    
    print(f"Loaded {len(df)} detector correlations")
    print(f"Correlation_ma range: [{df['correlation_ma'].min():.3f}, {df['correlation_ma'].max():.3f}]")
        
    return df

def generate_city_mapping(all_cities_list):
    mapping = { city: city.split(',')[0].lower() for city in all_cities_list }
    if "Los Angeles, USA" in mapping:
        mapping["Los Angeles, USA"] = "losanageles" # there is a typo in the detectors_public.csv file
    if "Lucerne, Switzerland" in mapping:
        mapping["Lucerne, Switzerland"] = "luzern"
    if "臺北市, Taiwan" in mapping:
        mapping["臺北市, Taiwan"] = "taipeh"
    if "東京23区, Japan" in mapping:
        mapping["東京23区, Japan"] = "tokyo"
    return mapping

all_cities_list = [
    "Augsburg, Germany",
    "Basel, Switzerland",
    "Bern, Switzerland",
    "Birmingham, UK",
    "Bolton, UK",
    "Bremen, Germany",
    "Bordeaux, France",
    "Cagliari, Italy",
    "Constance, Germany",
    "Darmstadt, Germany",
    "Essen, Germany",
    "Frankfurt, Germany",
    "Graz, Austria",
    "Groningen, Netherlands",
    "Hamburg, Germany",
    "Innsbruck, Austria",
    "Kassel, Germany",
    "London, UK",
    "Los Angeles, USA",
    "Lucerne, Switzerland",
    "Madrid, Spain",
    "Melbourne, Australia",
    "Manchester, UK",
    "Marseille, France",
    "Munich, Germany",
    "Paris, France",
    "Rotterdam, Netherlands",
    "Santander, Spain",
    "Speyer, Germany",
    "Strasbourg, France",
    "Stuttgart, Germany",
    "臺北市, Taiwan",
    "東京23区, Japan",
    "Torino, Italy",
    "Toulouse, France",
    "Utrecht, Netherlands",
    "Vilnius, Lithuania",
    "Wolfsburg, Germany",
    "Zurich, Switzerland",
]

def line_graph_to_osmnx_primal(G_roads):
    """
    Reconstruct a MultiDiGraph whose edges/edge-attrs match the original graph's edges,
    assuming G_roads is a line graph built from an OSMnx MultiDiGraph and its nodes are (u, v, k).
    
    Note: original *node* attributes (x, y, street_count, etc.) are NOT recoverable unless you saved them.
    """
    G = nx.MultiDiGraph()

    # (optional) restore graph-level attributes if you saved them
    if "orig_graph_attrs" in G_roads.graph:
        G.graph.update(G_roads.graph["orig_graph_attrs"])

    # add nodes from endpoints seen in line-graph node labels
    for (u, v, k) in G_roads.nodes:
        G.add_node(u)
        G.add_node(v)

    # (optional) restore node attributes if you saved them
    if "orig_node_attrs" in G_roads.graph:
        nx.set_node_attributes(G, G_roads.graph["orig_node_attrs"])

    # each line-graph node becomes an edge in the primal graph
    for (u, v, k), attrs in G_roads.nodes(data=True):
        G.add_edge(u, v, key=k, **dict(attrs))

    if "crs" in G_roads.graph:
        G.graph["crs"] = G_roads.graph["crs"]
    else:
        G.graph["crs"] = "epsg:4326"  # default to WGS84

    return G

def create_road_graphs_with_labels(
    detector_coords_file = os.path.join(SCRIPT_DIR, "..", "data", "traffic_data", "detectors_public.csv"),
    correlation_file = os.path.join(SCRIPT_DIR, "..", "analyse_utd19", "all_correlations", "correlations.csv"),
    cities = all_cities_list,
):    
    
    detectors_df = pd.read_csv(detector_coords_file)
    print(f"Loaded {len(detectors_df)} detector coordinates")
    
    correlations_df = load_correlation_data(correlation_file)
    
    # Merge coordinates with correlations
    # detector_id from correlations.csv should match detid from detectors_public.csv
    # Drop any detector ID that appears in one DataFrame but not the other.
    detector_data = detectors_df.merge(
        correlations_df,
        left_on='detid',
        right_on='detector_id',
        how='inner'
    )
    
    print(f"\nMerged data: {len(detector_data)} detectors with both location and correlation")
    
    if len(detector_data) == 0:
        print("ERROR: No detectors matched between files!")
        print("Check if 'detid' in detectors_public.csv matches 'detector_id' in correlations.csv")
        return {}
    
    # City name to code mapping
    city_mapping = generate_city_mapping(all_cities_list)

    results = {}
    for city in cities:
        print(f"\n{'='*60}")
        print(f"Processing {city}")
        print(f"{'='*60}")
        
        # Get city code for filtering
        city_code = city_mapping.get(city, city.split(',')[0].lower())
        
        # Filter detectors for this city
        city_detector_data = detector_data[
            detector_data['citycode'].str.lower() == city_code
        ].copy()
        
        if len(city_detector_data) == 0:
            print(f"Warning: No detectors found for {city}")
            continue
        
        print(f"Found {len(city_detector_data)} detectors with correlations in {city}")
        
        # LOAD GRAPH WITH POPULATION DENSITY (LINE GRAPH)
        print("Loading road graph from gpickle file wtih population density...")
        og_graph_file_path = os.path.join(SCRIPT_DIR, "..","data", "graphs", f"{city.replace(', ', '_').replace(' ', '_')}_road_graph_with_popdensity.gpickle")
        with open(og_graph_file_path, "rb") as f:
            og_graph_with_popdensity = pickle.load(f)

        # CREATE PRIMAL GRAPH FROM LINE GRAPH
        G = line_graph_to_osmnx_primal(og_graph_with_popdensity)

        # # Download street network from OpenStreetMap
        # print("Downloading road network from OSM...")
        # G = ox.graph_from_place(city, network_type="drive")
        # print(f"Original graph: {len(G.nodes)} nodes, {len(G.edges)} edges")
        
        # # Create line graph: edges become nodes (road segments)
        # print("Creating line graph (road-centric)...")
        # G_roads = nx.line_graph(G)
        # print(f"Line graph: {len(G_roads.nodes)} road segments (nodes)")
        
        # # Copy edge attributes from original graph to line graph nodes
        # for u, v, k, data in G.edges(keys=True, data=True):
        #     lg_node = (u, v, k)
        #     if lg_node in G_roads:
        #         G_roads.nodes[lg_node].update(data)
        
        # Map detectors to road segments
        print("Mapping detectors to road segments...")
        detector_long = city_detector_data['long'].values
        detector_lat = city_detector_data['lat'].values
        detector_ids = city_detector_data['detid'].values
        correlations = city_detector_data['correlation_ma'].values
        
        # Find nearest edge in original graph for each detector
        nearest_edges = ox.distance.nearest_edges(
            G, detector_long, detector_lat, return_dist=False
        )
        
        # Create mappings
        detector_to_road = {}  # detector_id -> road_node_id
        road_to_correlations = defaultdict(list)  # road_node_id -> [correlations]
        
        for detid, edge, corr in zip(detector_ids, nearest_edges, correlations):
            if edge in og_graph_with_popdensity.nodes:
                detector_to_road[detid] = edge
                road_to_correlations[edge].append(corr)
        
        print(f"Mapped {len(detector_to_road)} detectors to road segments")
        print(f"{len(road_to_correlations)} unique road segments have detectors")
        
        # Create label tensor (y) for all nodes
        print("\nCreating label tensor...")
        node_list = list(og_graph_with_popdensity.nodes())
        y = np.full(len(node_list), 0, dtype=np.float32)
        
        had_corr = 0   
        for idx, node in enumerate(node_list):
            if node in road_to_correlations:
                # If multiple detectors on same road, assign the minimum correlation value on the given road segment because it is most likely to show congestion
                min_corr = np.min(road_to_correlations[node])
                y[idx] = min_corr
                had_corr += 1
            else:
                # No detector --> value 0 gives a neutral signal
                y[idx] = 0.0 # Unlabeled
        
        # Convert to PyTorch tensor
        y_tensor = torch.tensor(y, dtype=torch.float32)
        
        # Print label statistics
        print(f"\nLabel statistics for {city}: {had_corr}/{len(node_list)} road segments have been assigned their correlation")

        # Sanity check: node sets should match between G_roads and pop-density graph
        # nodes_roads = set(G_roads.nodes())
        # nodes_pop = set(og_graph_with_popdensity.nodes())

        # missing_in_pop = nodes_roads - nodes_pop
        # missing_in_roads = nodes_pop - nodes_roads

        # print(f"|G_roads\\G_pop_density| = {len(missing_in_pop)}; |G_pop_density\\G_roads| = {len(missing_in_roads)}")

        # if missing_in_pop or missing_in_roads:
        #     print("WARNING: Node sets differ between line graph and pop-density graph!")

        # Store results
        results[city] = {
            'graph': og_graph_with_popdensity,
            'y': y_tensor,
            'node_list': node_list,
            'detector_mapping': detector_to_road,
        }
    
    return results

def calc_curvature(node_data):
    """
    Calculate curvature for a node based on its geometry. The curvature is defined as the ratio of the path length
    along the polyline to the straight-line distance between the endpoints, normalized so that straight lines have curvature 0.
    The curvature is clamped to [0, 5] and then log-scaled.

    Args:
        node_data (dict): Node data containing a 'geometry' key with a shapely LineString.
    Returns:
        float: Curvature value in range [0, log(6)].
    """
    default_return = 0

    geom = node_data.get("geometry")
    if geom is None or geom.is_empty:
        raise ValueError("Geometry is missing or empty")
        return default_return

    coords = list(geom.coords)
    if len(coords) < 2:
        raise ValueError("Geometry has fewer than 2 coordinates")
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

    return curvature

def save_graphs_and_labels(
    results: Dict[str, Dict],
    output_dir: str = "processed_graphs"
):
    """
    Save processed graphs and y tensors to disk for later training.
    
    Args:
        results: Output from create_road_graphs_with_labels
        output_dir: Directory to save files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for city, data in results.items():
        city_name = city.replace(',', '').replace(' ', '_')
        
        # Add correlation as node attribute
        for i, n in enumerate(data['node_list']):
            data['graph'].nodes[n]['correlation'] = data['y'][i].item()
        print(f"Added correlation values to graph nodes.")
        
        # Add curvature as node attribute
        for n in data['graph'].nodes:
            data['graph'].nodes[n]['curvature'] = calc_curvature(data['graph'].nodes[n])
        print(f"Added curvature values to graph nodes.")

        # keep only the desired node attributes
        desired_attributes = ['highway', 'length', 'lanes', 'maxspeed', 'pop_density', 'correlation', 'curvature']
        for n in data['graph'].nodes:
            node_attrs = data['graph'].nodes[n]
            for attr in list(node_attrs.keys()):
                if attr not in desired_attributes:
                    del node_attrs[attr]
            # if lanes is missing, set to 1, if it is present values can be like '2' (should convert to int) or ['2', '3'] (take average)
            if 'lanes' not in node_attrs:
                node_attrs['lanes'] = 1.0
            else:
                lanes_value = node_attrs['lanes']
                if isinstance(lanes_value, list):
                    try:
                        lanes_numbers = [float(lane) for lane in lanes_value]
                        node_attrs['lanes'] = float(np.mean(lanes_numbers))
                    except:
                        node_attrs['lanes'] = 1.0
                else:
                    try:
                        node_attrs['lanes'] = float(lanes_value)
                    except:
                        node_attrs['lanes'] = 1.0
            if 'highway' not in node_attrs:
                node_attrs['highway'] = 'residential'
            if 'length' not in node_attrs:
                node_attrs['length'] = 0.0
            if 'maxspeed' not in node_attrs:
                node_attrs['maxspeed'] = 50.0
            else:
                maxspeed_value = node_attrs['maxspeed']
                if isinstance(maxspeed_value, list):
                    try:
                        speeds = [float(speed) for speed in maxspeed_value]
                        node_attrs['maxspeed'] = float(np.mean(speeds))
                    except:
                        node_attrs['maxspeed'] = 50.0
                else:
                    try:
                        node_attrs['maxspeed'] = float(maxspeed_value)
                    except:
                        node_attrs['maxspeed'] = 50.0
            if 'pop_density' not in node_attrs:
                node_attrs['pop_density'] = 0.0
            if 'correlation' not in node_attrs:
                node_attrs['correlation'] = 0.0
            if 'curvature' not in node_attrs:
                node_attrs['curvature'] = 0.0
        
        # check the size of the graph
        print("Saving data for city:", city)
        print(f"- Graph has {len(data['graph'].nodes)} nodes and {len(data['graph'].edges)} edges")

        # check whether all nodes have the same attributes (by printing the list of attributes of 10 random nodes)
        attribute_list = list(data['graph'].nodes(data=True))[0][1].keys()
        for i in range(3):
            random_node = list(data['graph'].nodes())[i*50]
            print(f"  - Node {random_node} has attributes: {sorted(list(data['graph'].nodes[random_node].keys()))}")

        # Save graph (NetworkX format) using pickle
        # graph_file = os.path.join(output_dir, f"{city_name}_graph.gpickle")
        # with open(graph_file, 'wb') as f:
        #     pickle.dump(data['graph'], f, pickle.HIGHEST_PROTOCOL)
        graph_zipped_file = os.path.join(output_dir, f"{city_name}_graph.pkl.gz")
        with gzip.open(graph_zipped_file, "wb") as f:
            pickle.dump(data['graph'], f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Saved graph: {graph_zipped_file} to {graph_zipped_file}")
        
    print(f"\nAll data saved to {output_dir}/")


if __name__ == "__main__":
    print("="*70)
    print("CREATING ROAD GRAPHS WITH CONGESTION LABELS (Y TENSOR)")
    print("="*70)
    try:
        # Create graphs with y labels
        results = create_road_graphs_with_labels(
            detector_coords_file=os.path.join(SCRIPT_DIR, "..", "data", "traffic_data", "detectors_public.csv"),
            correlation_file=os.path.join(SCRIPT_DIR, "..", "analyse_utd19", "all_correlations", "correlations.csv"),
            # cities=["Graz, Austria"], # For testing, process only one city
            cities=all_cities_list
            # cities=[all_cities_list[0]], # For testing, process only one city
        )
        
        if not results:
            print("\nERROR: No results generated. Check your input files.")
        else:
            # Save all data to disk
            print("\n" + "="*70)
            print("SAVING DATA")
            print("="*70)
            save_graphs_and_labels(results, output_dir=os.path.join(SCRIPT_DIR, "..", "data", "final_graphs"))
            
            # Show summary
            print("\n" + "="*70)
            print("SUMMARY")
            print("="*70)
            for city, data in results.items():
                print(f"\n{city}:")
                print(f"  Graph nodes (road segments): {len(data['graph'].nodes)}")
                print(f"  Graph edges: {len(data['graph'].edges)}")
            
            print("\n" + "="*70)
            print("✓ Done! Ready for GNN training.")
            print("="*70)
            
    except FileNotFoundError as e:
        print(f"\n ERROR: File not found - {e}")
        print("Make sure 'detectors_public.csv' and 'correlations.csv' are in the same directory.")
    except Exception as e:
        print(f"\n ERROR: {e}")
        import traceback
        traceback.print_exc()