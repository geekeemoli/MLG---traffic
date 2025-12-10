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
    
    # Select only detector_id and correlation_ma
    df = df[['detector_id', 'correlation_ma']].copy()
    
    # Remove any NaN correlations
    df = df.dropna(subset=['correlation_ma'])
    
    print(f"Loaded {len(df)} detector correlations")
    print(f"Correlation_ma range: [{df['correlation_ma'].min():.3f}, {df['correlation_ma'].max():.3f}]")
    
    # Show distribution
    congestion = (df['correlation_ma'] < -0.1).sum()
    no_congestion = (df['correlation_ma'] > 0.1).sum()
    dead_zone = ((df['correlation_ma'] >= -0.1) & (df['correlation_ma'] <= 0.1)).sum()
    
    print(f"Distribution in correlations.csv:")
    print(f"  Congestion (< -0.1): {congestion}")
    print(f"  No congestion (> 0.1): {no_congestion}")
    print(f"  Dead zone (-0.1 to 0.1): {dead_zone}")
    
    return df

def generate_city_mapping(raw_city_data: str) -> Dict[str, str]:
    """
    Converts a string of city and country data into a city code mapping dictionary.

    The input string must use a semicolon (;) as a separator and have a header row.
    
    Args:
        raw_city_data: A string containing the city data (e.g., "City;Country\nAugsburg;Germany").

    Returns:
        A dictionary where:
        - Keys are "City, Country" (e.g., "Augsburg, Germany")
        - Values are the lowercase city name code (e.g., "augsburg")
    """
    city_mapping = {}
    
    # 1. Use StringIO to treat the string as a file object
    # This allows the csv module to read the data line by line
    data_file = StringIO(raw_city_data)
    
    # 2. Use csv.reader to parse the data based on the delimiter
    # It automatically handles newline characters (\n)
    reader = csv.reader(data_file, delimiter=';')
    
    # Skip the header row (e.g., "City;Country")
    next(reader) 
    
    # 3. Iterate through the remaining rows and build the dictionary
    for row in reader:
        # Skip empty rows if any
        if not row:
            continue
            
        # The list row will contain [CityName, CountryName]
        city_name = row[0].strip()
        country_name = row[1].strip()
        
        # Create the dictionary key: "City, Country"
        dict_key = f"{city_name}, {country_name}"
        
        # Create the dictionary value (the code): lowercase city name, 
        # removing spaces for a clean code
        dict_value = city_name.lower().replace(" ", "")
        
        city_mapping[dict_key] = dict_value
        
    return city_mapping

# --- YOUR INPUT DATA ---
# It's important to use triple quotes (""") to handle multi-line strings easily.

raw_city_list = """
Augsburg;Germany
Basel;Switzerland
Berne;Switzerland
Birmingham;UK
Bolton;UK
Bremen;Germany
Bordeaux;France
Cagliari;Italy
Constance;Germany
Darmstadt;Germany
Essen;Germany
Frankfurt;Germany
Graz;Austria
Groningen;Netherlands
Hamburg;Germany
Innsbruck;Austria
Kassel;Germany
London;UK
Los Angeles;USA
Lucerne;Switzerland
Madrid;Spain
Melbourne;Australia
Manchester;UK
Marseille;France
Munich;Germany
Paris;France
Rotterdam;Netherlands
Santander;Spain
Speyer;Germany
Strasbourg;France
Stuttgart;Germany
Taipei;Taiwan
Tokyo;Japan
Torino;Italy
Toulouse;France
Utrecht;Netherlands
Vilnius;Lithuania
Wolfsburg;Germany
Zurich;Switzerland
"""

def generate_city_name_list(raw_city_data: str) -> str:
    """
    Converts a string of city and country data into a comma-separated list of 
    full names, enclosed in double quotes (e.g., "City 1, Country 1", "City 2, Country 2").
    
    Args:
        raw_city_data: A string containing the city data (e.g., "City;Country\nAugsburg;Germany").

    Returns:
        A formatted string of city and country names.
    """
    full_names = []
    
    # Use StringIO to treat the string as a file object
    data_file = StringIO(raw_city_data)
    
    # Use csv.reader to parse the data based on the delimiter
    reader = csv.reader(data_file, delimiter=';')
    
    # Skip the header row
    try:
        next(reader) 
    except StopIteration:
        # Handle case where data is empty
        return ""
    
    # Iterate through the remaining rows and format the names
    for row in reader:
        # Skip empty rows if any
        if len(row) < 2:
            continue
            
        city_name = row[0].strip()
        country_name = row[1].strip()
        
        # Format as "City, Country"
        full_name = f'"{city_name}, {country_name}"'
        full_names.append(full_name)
        
    # Join the list elements with ", "
    return ", ".join(full_names)

string_of_cities = generate_city_name_list(raw_city_list.strip())

def create_road_graphs_with_labels(
    detector_coords_file: str = "data/traffic_data/detectors_public.csv",
    correlation_file: str = "analyse_utd19/all_correlations/correlations.csv",
    cities: List[str] = [string_of_cities],
    threshold_low: float = -0.1,
    threshold_high: float = 0.1,
    unlabeled_value: float = 0.0
) -> Dict[str, Dict]:
    """
    Create line graphs for cities and assign congestion labels based on detector correlations.
    
    Labels (y tensor values):
        - correlation_ma < -0.3: Label = 1 (CONGESTION/CRITICAL)
        - correlation_ma > 0.3: Label = 0 (NO CONGESTION)
        - -0.3 <= correlation_ma <= 0.3: Label = 0 (UNLABELED - dead zone)
        - No detector: Label = 0 (UNLABELED)
    
    Returns:
        Dict mapping city name to:
            - 'graph': nx.Graph (line graph of roads)
            - 'y': torch.Tensor of shape (num_nodes,) - labels for supervised learning
            - 'node_list': List of node IDs in same order as y tensor
            - 'detector_mapping': Dict[detector_id -> road_node_id]
            - 'stats': Label statistics
    """
    
    # Load detector coordinates
    detectors_df = pd.read_csv(detector_coords_file)
    print(f"Loaded {len(detectors_df)} detector coordinates")
    
    # Load correlations
    correlations_df = load_correlation_data(correlation_file)
    
    # Merge coordinates with correlations
    # detector_id from correlations.csv should match detid from detectors_public.csv
    #Drop any detector ID that appears in one DataFrame but not the other.
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
    city_mapping = generate_city_mapping(raw_city_list.strip())

    
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
        
        # Download street network from OpenStreetMap
        print("Downloading road network from OSM...")
        G = ox.graph_from_place(city, network_type="drive")
        print(f"Original graph: {len(G.nodes)} nodes, {len(G.edges)} edges")
        
        # Create line graph: edges become nodes (road segments)
        print("Creating line graph (road-centric)...")
        G_roads = nx.line_graph(G)
        print(f"Line graph: {len(G_roads.nodes)} road segments (nodes)")
        
        # Copy edge attributes from original graph to line graph nodes
        for u, v, k, data in G.edges(keys=True, data=True):
            lg_node = (u, v, k)
            if lg_node in G_roads:
                G_roads.nodes[lg_node].update(data)
        
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
            if edge in G_roads.nodes:
                detector_to_road[detid] = edge
                road_to_correlations[edge].append(corr)
        
        print(f"Mapped {len(detector_to_road)} detectors to road segments")
        print(f"{len(road_to_correlations)} unique road segments have detectors")
        
        # Create label tensor (y) for all nodes
        print("\nCreating label tensor...")
        node_list = list(G_roads.nodes())
        y = np.full(len(node_list), unlabeled_value, dtype=np.float32)
        
        stats = {
            'congestion': 0,
            'no_congestion': 0,
            'unlabeled_dead_zone': 0,
            'unlabeled_no_detector': 0
        }
        
        for idx, node in enumerate(node_list):
            if node in road_to_correlations:
                # If multiple detectors on same road, average their correlations
                avg_corr = np.mean(road_to_correlations[node])
                
                if avg_corr < threshold_low:
                    y[idx] = 1.0  # Congestion (critical road)
                    stats['congestion'] += 1
                elif avg_corr > threshold_high:
                    y[idx] = 0.0  # No congestion
                    stats['no_congestion'] += 1
                else:
                    # In dead zone (-0.3 to 0.3)
                    y[idx] = unlabeled_value
                    stats['unlabeled_dead_zone'] += 1
            else:
                # No detector on this road segment
                y[idx] = unlabeled_value
                stats['unlabeled_no_detector'] += 1
        
        # Convert to PyTorch tensor
        y_tensor = torch.tensor(y, dtype=torch.float32)
        
        # Print label statistics
        print(f"\nLabel statistics for {city}:")
        print(f"  Congestion (y=1): {stats['congestion']} ({100*stats['congestion']/len(node_list):.2f}%)")
        print(f"  No congestion (y=0): {stats['no_congestion']} ({100*stats['no_congestion']/len(node_list):.2f}%)")
        print(f"  Unlabeled (dead zone): {stats['unlabeled_dead_zone']} ({100*stats['unlabeled_dead_zone']/len(node_list):.2f}%)")
        print(f"  Unlabeled (no detector): {stats['unlabeled_no_detector']} ({100*stats['unlabeled_no_detector']/len(node_list):.2f}%)")
        print(f"  Total road segments: {len(node_list)}")
        
        # Store results
        results[city] = {
            'graph': G_roads,
            'y': y_tensor,
            'node_list': node_list,
            'detector_mapping': detector_to_road,
            'stats': stats
        }
    
    return results


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
        
        # Save graph (NetworkX format) using pickle
        graph_file = os.path.join(output_dir, f"{city_name}_graph.gpickle")
        with open(graph_file, 'wb') as f:
            pickle.dump(data['graph'], f, pickle.HIGHEST_PROTOCOL)
        print(f"Saved graph: {graph_file}")
        
        # Save y tensor (labels)
        y_file = os.path.join(output_dir, f"{city_name}_y.pt")
        torch.save(data['y'], y_file)
        print(f"Saved y tensor: {y_file}")
        
        # Save node list (to maintain node ordering)
        node_file = os.path.join(output_dir, f"{city_name}_node_list.pkl")
        with open(node_file, 'wb') as f:
            pickle.dump(data['node_list'], f)
        print(f"Saved node list: {node_file}")
        
        # Save metadata
        meta_file = os.path.join(output_dir, f"{city_name}_metadata.pkl")
        metadata = {
            'stats': data['stats'],
            'detector_mapping': data['detector_mapping']
        }
        with open(meta_file, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"Saved metadata: {meta_file}")
        
    print(f"\nAll data saved to {output_dir}/")


def load_city_data(city_name: str, data_dir: str = "processed_graphs") -> Dict:
    """
    Load a previously saved city's graph and labels.
    
    Args:
        city_name: Name like "Graz_Austria" or "Munich_Germany"
        data_dir: Directory where data was saved
        
    Returns:
        Dict with 'graph', 'y', 'node_list', 'metadata'
    """
    with open(os.path.join(data_dir, f"{city_name}_graph.gpickle"), 'rb') as f:
        graph = pickle.load(f)
    y = torch.load(os.path.join(data_dir, f"{city_name}_y.pt"))
    
    with open(os.path.join(data_dir, f"{city_name}_node_list.pkl"), 'rb') as f:
        node_list = pickle.load(f)
    
    with open(os.path.join(data_dir, f"{city_name}_metadata.pkl"), 'rb') as f:
        metadata = pickle.load(f)
    
    return {
        'graph': graph,
        'y': y,
        'node_list': node_list,
        'metadata': metadata
    }


if __name__ == "__main__":
    print("="*70)
    print("CREATING ROAD GRAPHS WITH CONGESTION LABELS (Y TENSOR)")
    print("="*70)
    cities_for_input = generate_city_name_list(raw_city_list.strip())
    try:
        # Create graphs with y labels
        results = create_road_graphs_with_labels(
            detector_coords_file="data/traffic_data/detectors_public.csv",
            correlation_file="analyse_utd19/all_correlations/correlations.csv",
            cities=["Graz, Austria", "Munich, Germany", "Zurich, Switzerland", "Wolfsburg, Germany"],
            threshold_low=-0.1,
            threshold_high=0.1,
            unlabeled_value=0.0
        )
        
        if not results:
            print("\nERROR: No results generated. Check your input files.")
        else:
            # Save all data to disk
            print("\n" + "="*70)
            print("SAVING DATA")
            print("="*70)
            save_graphs_and_labels(results, output_dir="processed_graphs")
            
            # Show summary
            print("\n" + "="*70)
            print("SUMMARY")
            print("="*70)
            for city, data in results.items():
                print(f"\n{city}:")
                print(f"  Graph nodes (road segments): {len(data['graph'].nodes)}")
                print(f"  Graph edges: {len(data['graph'].edges)}")
                print(f"  Y tensor shape: {data['y'].shape}")
                print(f"  Congestion labels (y=1): {(data['y'] == 1).sum().item()}")
                print(f"  No-congestion labels (y=0 labeled): {data['stats']['no_congestion']}")
                print(f"  Unlabeled (y=0 unlabeled): {data['stats']['unlabeled_dead_zone'] + data['stats']['unlabeled_no_detector']}")
            
            print("\n" + "="*70)
            print("✓ Done! Ready for GNN training.")
            print("="*70)
            
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: File not found - {e}")
        print("Make sure 'detectors_public.csv' and 'correlations.csv' are in the same directory.")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()