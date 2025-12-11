import torch
import networkx as nx
from torch_geometric.data import Data
from typing import Dict, List, Optional, Tuple
import numpy as np


def networkx_to_pyg(
    graph: nx.Graph,
    node_list: List,
    y_tensor: Optional[torch.Tensor] = None,
    node_features: Optional[List[str]] = None,
    use_default_features: bool = True
) -> Data:
    """
    Convert a NetworkX graph to PyTorch Geometric Data object.
    
    Args:
        graph: NetworkX graph (line graph of road segments)
        node_list: Ordered list of node IDs (must match y_tensor ordering)
        y_tensor: Target labels tensor of shape (num_nodes,). If None, uses 'correlation' attribute
        node_features: List of node attribute names to use as features
        use_default_features: If True and node_features is None, uses common road attributes
        
    Returns:
        PyTorch Geometric Data object with:
            - x: Node feature matrix [num_nodes, num_features]
            - edge_index: Graph connectivity [2, num_edges]
            - y: Target labels [num_nodes] (if provided)
            - node_mapping: Dict mapping node_id -> index in tensor
    """
    
    # Create node index mapping
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}
    num_nodes = len(node_list)
    
    # === EDGE INDEX ===
    # Convert edges to tensor format [2, num_edges]
    edge_list = []
    for u, v in graph.edges():
        if u in node_to_idx and v in node_to_idx:
            edge_list.append([node_to_idx[u], node_to_idx[v]])
            edge_list.append([node_to_idx[v], node_to_idx[u]])  # Undirected graph
    
    if len(edge_list) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    print(f"Created edge_index: {edge_index.shape}")
    
    # === NODE FEATURES ===
    # Determine which features to extract
    if node_features is None and use_default_features:
        # Common road network attributes from OSM
        node_features = [
            'length',           # Road segment length
            'lanes',            # Number of lanes
            'maxspeed',         # Speed limit
            'highway',          # Road type (encoded)
            'population_density'  # From your population density data
        ]
    elif node_features is None:
        node_features = []
    
    # Extract features for each node
    feature_matrix = []
    
    for node in node_list:
        node_data = graph.nodes[node]
        features = []
        
        # Length feature
        if 'length' in node_features:
            length = node_data.get('length', 0.0)
            features.append(float(length))
        
        # Lanes feature
        if 'lanes' in node_features:
            lanes = node_data.get('lanes', 1)
            if isinstance(lanes, (list, tuple)):
                lanes = lanes[0] if len(lanes) > 0 else 1
            try:
                lanes = float(lanes)
            except (ValueError, TypeError):
                lanes = 1.0
            features.append(lanes)
        
        # Maxspeed feature
        if 'maxspeed' in node_features:
            maxspeed = node_data.get('maxspeed', 50)
            if isinstance(maxspeed, str):
                try:
                    maxspeed = float(maxspeed.split()[0])  # Handle "50 mph" format
                except (ValueError, AttributeError):
                    maxspeed = 50.0
            elif isinstance(maxspeed, (list, tuple)):
                maxspeed = maxspeed[0] if len(maxspeed) > 0 else 50.0
            features.append(float(maxspeed))
        
        # Highway type (encode as categorical)
        if 'highway' in node_features:
            highway_types = {
                'motorway': 6, 'trunk': 5, 'primary': 4,
                'secondary': 3, 'tertiary': 2, 'residential': 1,
                'unclassified': 0
            }
            highway = node_data.get('highway', 'unclassified')
            if isinstance(highway, (list, tuple)):
                highway = highway[0] if len(highway) > 0 else 'unclassified'
            highway_code = highway_types.get(highway, 0)
            features.append(float(highway_code))
        
        # Population density
        if 'population_density' in node_features:
            pop_density = node_data.get('population_density', 0.0)
            features.append(float(pop_density))
        
        # Add any custom features
        for feat_name in node_features:
            if feat_name not in ['length', 'lanes', 'maxspeed', 'highway', 'population_density']:
                feat_value = node_data.get(feat_name, 0.0)
                features.append(float(feat_value))
        
        feature_matrix.append(features)
    
    # Convert to tensor
    if len(feature_matrix) > 0 and len(feature_matrix[0]) > 0:
        x = torch.tensor(feature_matrix, dtype=torch.float)
        print(f"Created feature matrix x: {x.shape}")
        
        # Normalize features (min-max scaling per feature)
        for i in range(x.shape[1]):
            col = x[:, i]
            min_val = col.min()
            max_val = col.max()
            if max_val > min_val:
                x[:, i] = (col - min_val) / (max_val - min_val)
    else:
        # No features available - use one-hot encoding of node degree
        degrees = [graph.degree(node) for node in node_list]
        x = torch.tensor(degrees, dtype=torch.float).unsqueeze(1)
        print(f"No features found, using node degrees: {x.shape}")
    
    # === LABELS ===
    # Use provided y_tensor or extract from 'correlation' attribute
    if y_tensor is not None:
        y = y_tensor
    elif 'correlation' in graph.nodes[node_list[0]]:
        y = torch.tensor([graph.nodes[node].get('correlation', 0.0) for node in node_list], 
                        dtype=torch.float)
    else:
        y = None
    
    # Create PyG Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        num_nodes=num_nodes
    )
    
    # Store node mapping for reference
    data.node_mapping = node_to_idx
    data.node_list = node_list
    
    return data


def batch_networkx_to_pyg(
    results: Dict[str, Dict],
    node_features: Optional[List[str]] = None,
    use_default_features: bool = True
) -> Dict[str, Data]:
    """
    Convert multiple city graphs to PyTorch Geometric format.
    
    Args:
        results: Output from create_road_graphs_with_labels
        node_features: List of node attribute names to use as features
        use_default_features: If True, uses default road network features
        
    Returns:
        Dictionary mapping city name to PyG Data object
    """
    pyg_data = {}
    
    for city, data in results.items():
        print(f"\n{'='*60}")
        print(f"Converting {city} to PyTorch Geometric format")
        print(f"{'='*60}")
        
        pyg_graph = networkx_to_pyg(
            graph=data['graph'],
            node_list=data['node_list'],
            y_tensor=data['y'],
            node_features=node_features,
            use_default_features=use_default_features
        )
        
        pyg_data[city] = pyg_graph
        
        print(f"\nPyG Data summary for {city}:")
        print(f"  Number of nodes: {pyg_graph.num_nodes}")
        print(f"  Number of edges: {pyg_graph.edge_index.shape[1]}")
        print(f"  Node feature dim: {pyg_graph.x.shape[1]}")
        if pyg_graph.y is not None:
            print(f"  Label tensor shape: {pyg_graph.y.shape}")
            print(f"  Label range: [{pyg_graph.y.min():.3f}, {pyg_graph.y.max():.3f}]")
    
    return pyg_data


def save_pyg_data(
    pyg_data: Dict[str, Data],
    output_dir: str = "data/pyg_graphs"
):
    """
    Save PyTorch Geometric Data objects to disk.
    
    Args:
        pyg_data: Dictionary mapping city name to PyG Data object
        output_dir: Directory to save files
    """
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    for city, data in pyg_data.items():
        city_name = city.replace(',', '').replace(' ', '_')
        output_file = os.path.join(output_dir, f"{city_name}_pyg.pt")
        
        torch.save(data, output_file)
        print(f"Saved PyG data for {city}: {output_file}")
    
    print(f"\nAll PyG data saved to {output_dir}/")


def load_pyg_data(city_name: str, data_dir: str = "data/pyg_graphs") -> Data:
    """
    Load a previously saved PyG Data object.
    
    Args:
        city_name: Name like "Graz_Austria" or "Munich_Germany"
        data_dir: Directory where data was saved
        
    Returns:
        PyTorch Geometric Data object
    """
    import os
    
    file_path = os.path.join(data_dir, f"{city_name}_pyg.pt")
    data = torch.load(file_path)
    
    print(f"Loaded PyG data for {city_name}:")
    print(f"  Nodes: {data.num_nodes}")
    print(f"  Edges: {data.edge_index.shape[1]}")
    print(f"  Features: {data.x.shape}")
    
    return data


# Example usage
if __name__ == "__main__":
    """
    Example of how to use these functions with your existing code.
    
    After running create_road_graphs_with_labels(), you can convert to PyG format:
    
    # Convert all graphs to PyG format
    pyg_data = batch_networkx_to_pyg(results)
    
    # Save PyG data
    save_pyg_data(pyg_data, output_dir="data/pyg_graphs")
    
    # Later, load for training
    graz_data = load_pyg_data("Graz_Austria", data_dir="data/pyg_graphs")
    """
    print("NetworkX to PyTorch Geometric converter ready!")
    print("\nUsage:")
    print("1. Run create_road_graphs_with_labels() to get results dict")
    print("2. Call batch_networkx_to_pyg(results) to convert all graphs")
    print("3. Use save_pyg_data() to save for later training")