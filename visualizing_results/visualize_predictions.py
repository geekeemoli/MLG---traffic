"""
Visualize model predictions vs actual detector data for London.
Creates a map showing predicted traffic jams vs actual measurements.
"""

import torch
import torch.nn as nn
import gzip
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torch_geometric.utils import from_networkx
from shapely.geometry import LineString, Point


CITY = "Graz_Austria"
MODEL_PATH = "../train/result_gat_ensemble_x5/model.pth"
GRAPH_PATH = f"../data/final_graphs/{CITY}_graph.pkl.gz"
GEOM_GRAPH_PATH = f"../data/graphs/{CITY}_road_graph_with_popdensity.gpickle"

CITY_CENTER = {
    "west": 15.35,
    "east": 15.52,
    "south": 47.01,
    "north": 47.12
}

MODEL_TYPE = "gat_ensemble"
N_MODELS = 5 
NUM_HIGHWAY_CLASSES = 7
HIDDEN_DIM = 64
NUM_LAYERS = 3
NUM_HEADS = 4
DROPOUT = 0

def load_graph_nx(path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)

def load_geom_graph(path):
    """Load intermediate graph with geometry (not compressed)."""
    with open(path, "rb") as f:
        return pickle.load(f)

def load_graph_pyg(path):
    graph_nx = load_graph_nx(path)
    graph_pyg = from_networkx(graph_nx)
    return graph_pyg, graph_nx

class FeatureBuilder(nn.Module):
    def __init__(self, num_highway_classes: int):
        super().__init__()
        all_highway_types = ["residential", "tertiary", "unclassified", "secondary", "primary", "trunk", "living_street", "motorway_link", "primary_link", "trunk_link", "motorway", "secondary_link", "tertiary_link", "busway", "crossing", "road", "escape", "yes", "via_ferrata", "alley", "emergency_bay"]
        self.highway_types = all_highway_types[:num_highway_classes]

    def forward(self, data):
        num_feats = torch.stack([
            data.curvature,
            data.lanes,
            data.length,
            data.maxspeed,
            data.pop_density,
        ], dim=-1).float()

        highway_lists = [[v] if isinstance(v, str) else v for v in data.highway]
        multi_hot_highway = torch.zeros((data.num_nodes, len(self.highway_types)+1), dtype=torch.float)
        for node_idx, highway_list in enumerate(highway_lists):
            for h in highway_list:
                if h in self.highway_types:
                    h_idx = self.highway_types.index(h)
                else:
                    h_idx = len(self.highway_types)
                multi_hot_highway[node_idx, h_idx] = 1.0

        data.x = torch.cat([num_feats, multi_hot_highway], dim=-1)
        return data

from torch_geometric.nn.models import GAT

class GATModelEnsemble(nn.Module):
    def __init__(self, n_models, in_channels, out_channels, hidden_channels=64, num_layers=3, heads=4, dropout=0.2):
        super(GATModelEnsemble, self).__init__()
        
        self.models = nn.ModuleList([
            GAT(
                in_channels=in_channels,
                out_channels=out_channels,
                hidden_channels=hidden_channels,
                num_layers=num_layers,
                heads=heads,
                dropout=dropout,
                v2=True,
            ) for _ in range(n_models)
        ])

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        out = []
        for model in self.models:
            out.append(model(x, edge_index).squeeze())
        out = torch.mean(torch.stack(out), dim=0)
        return out

class MLPBaselineModel(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, num_layers=3, dropout=0.2):
        super(MLPBaselineModel, self).__init__()
        
        layers = []
        layers.append(nn.Linear(in_channels, hidden_channels))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_channels, hidden_channels))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(hidden_channels, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, data):
        x = data.x
        out = self.mlp(x).squeeze()
        return out

def get_road_center(node_data):
    """Extract center coordinates from node geometry."""
    geom = node_data.get('geometry')
    if geom is not None:
        try:
            c = geom.centroid
            return float(c.x), float(c.y)
        except:
            pass
    return None, None

def get_road_coords(node_data):
    """Get full road coordinates for plotting."""
    geom = node_data.get('geometry')
    if geom is not None and hasattr(geom, 'coords'):
        coords = list(geom.coords)
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        return lons, lats
    return None, None

def is_in_bounds(lon, lat, bounds):
    """Check if coordinates are within bounding box."""
    if lon is None or lat is None:
        return False
    return (bounds["west"] <= lon <= bounds["east"] and 
            bounds["south"] <= lat <= bounds["north"])

def main():
    print(f"Loading graph from {GRAPH_PATH}...")
    graph_pyg, graph_nx = load_graph_pyg(GRAPH_PATH)
    print(f"Graph has {graph_pyg.num_nodes} nodes and {graph_pyg.num_edges} edges")
    
    print(f"Loading geometry from {GEOM_GRAPH_PATH}...")
    if os.path.exists(GEOM_GRAPH_PATH):
        geom_graph = load_geom_graph(GEOM_GRAPH_PATH)
        print(f"Geometry graph loaded with {geom_graph.number_of_nodes()} nodes")
    else:
        print(f"WARNING: Geometry graph not found at {GEOM_GRAPH_PATH}")
        print("Cannot visualize without geometry data.")
        return

    feature_builder = FeatureBuilder(num_highway_classes=NUM_HIGHWAY_CLASSES)
    graph_pyg = feature_builder(graph_pyg)

    geom_node_ids = list(geom_graph.nodes())
    sample_lons, sample_lats = [], []
    for node_id in geom_node_ids[:100]:
        node_data = geom_graph.nodes[node_id]
        lon, lat = get_road_center(node_data)
        if lon is not None:
            sample_lons.append(lon)
            sample_lats.append(lat)
    
    if sample_lons:
        print(f"\nDebug - Coordinate ranges in geometry graph:")
        print(f"  Longitude: {min(sample_lons):.4f} to {max(sample_lons):.4f}")
        print(f"  Latitude:  {min(sample_lats):.4f} to {max(sample_lats):.4f}")
    
    print(f"\nLoading model from {MODEL_PATH}...")
    in_channels = graph_pyg.x.size(-1)
    
    if MODEL_TYPE == "gat_ensemble":
        model = GATModelEnsemble(
            n_models=N_MODELS,
            in_channels=in_channels,
            out_channels=1,
            hidden_channels=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            heads=NUM_HEADS,
            dropout=DROPOUT
        )
    else:
        model = MLPBaselineModel(
            in_channels=in_channels,
            hidden_channels=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT
        )
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        print("Model loaded successfully!")
    else:
        print(f"WARNING: Model file not found at {MODEL_PATH}")
        print("Will visualize only ground truth data.")
        model = None
    
    if model is not None:
        model.eval()
        with torch.no_grad():
            logits = model(graph_pyg)
            predictions = torch.sigmoid(logits).numpy()
    else:
        predictions = None
    
    correlations = graph_pyg.correlation.numpy()
    
    node_ids = list(graph_nx.nodes())

    node_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}

    roads_data = []
    for node_id in geom_graph.nodes():
        geom_data = geom_graph.nodes[node_id]
        lon, lat = get_road_center(geom_data)
        
        if not is_in_bounds(lon, lat, CITY_CENTER):
            continue

        if node_id in node_to_idx:
            idx = node_to_idx[node_id]
            correlation = correlations[idx]
            prediction = predictions[idx] if predictions is not None else None
        else:
            correlation = 0
            prediction = None
        
        lons, lats = get_road_coords(geom_data)
        
        roads_data.append({
            'node_id': node_id,
            'lon': lon,
            'lat': lat,
            'lons': lons,
            'lats': lats,
            'correlation': correlation,
            'prediction': prediction,
            'has_detector': correlation != 0,
            'is_jam_actual': correlation < 0,
            'is_jam_predicted': prediction >= 0.5 if prediction is not None else None,
            'highway': geom_data.get('highway', 'unknown'),
        })
    
    print(f"\nFound {len(roads_data)} roads in {CITY.replace('_', ', ')} area")
    
    with_detector = [r for r in roads_data if r['has_detector']]
    actual_jams = [r for r in with_detector if r['is_jam_actual']]
    
    print(f"Roads with detector data: {len(with_detector)}")
    print(f"Actual traffic jams (correlation < 0): {len(actual_jams)}")
    
    if predictions is not None:
        predicted_jams = [r for r in roads_data if r['is_jam_predicted']]
        print(f"Predicted traffic jams: {len(predicted_jams)}")
        
        if len(with_detector) > 0:
            correct = sum(1 for r in with_detector if r['is_jam_actual'] == r['is_jam_predicted'])
            print(f"Accuracy on detector roads: {correct}/{len(with_detector)} = {100*correct/len(with_detector):.1f}%")
        else:
            print("No detector roads in this area to compute accuracy.")
    
    fig, axes = plt.subplots(1, 2 if predictions is not None else 1, figsize=(16 if predictions is not None else 8, 8))
    
    if predictions is None:
        axes = [axes]
    
    color_no_data = '#cccccc'
    color_no_jam = '#2ecc71'
    color_jam = '#e74c3c'       
    
    def plot_roads(ax, roads, use_predictions=False, title=""):
        ax.set_facecolor('#1a1a2e')
        
        for road in roads:
            if road['lons'] is None:
                continue
            
            if use_predictions:
                if road['prediction'] is None:
                    color = color_no_data
                    alpha = 0.3
                    lw = 0.5
                elif road['is_jam_predicted']:
                    color = color_jam
                    alpha = 0.9
                    lw = 1.5
                else:
                    color = color_no_jam
                    alpha = 0.7
                    lw = 1.0
            else:
                if not road['has_detector']:
                    color = color_no_data
                    alpha = 0.3
                    lw = 0.5
                elif road['is_jam_actual']:
                    color = color_jam
                    alpha = 0.9
                    lw = 1.5
                else:
                    color = color_no_jam
                    alpha = 0.7
                    lw = 1.0
            
            ax.plot(road['lons'], road['lats'], color=color, alpha=alpha, linewidth=lw)
        
        ax.set_xlim(CITY_CENTER['west'], CITY_CENTER['east'])
        ax.set_ylim(CITY_CENTER['south'], CITY_CENTER['north'])
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=color_jam, linewidth=2, label='Traffic Jam'),
            Line2D([0], [0], color=color_no_jam, linewidth=2, label='No Jam'),
            Line2D([0], [0], color=color_no_data, linewidth=1, alpha=0.5, label='No Data'),
        ]
        ax.legend(handles=legend_elements, loc='upper right')
    
    plot_roads(axes[0], roads_data, use_predictions=False, 
               title=f"Actual Traffic (from UTD19 Detectors)\n{len(with_detector)} roads with data, {len(actual_jams)} jams")
    
    if predictions is not None:
        predicted_jams_count = sum(1 for r in roads_data if r['is_jam_predicted'])
        plot_roads(axes[1], roads_data, use_predictions=True,
                   title=f"Model Predictions\n{predicted_jams_count} predicted jams")
    
    fig.suptitle(f'Traffic Comparison - {CITY.replace("_", ", ")}', fontsize=16, fontweight='bold', y=0.98)
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.05, right=0.98, wspace=0.15)
    
    output_path = f"{CITY}_traffic_comparison.png"
    plt.savefig(output_path, dpi=200, facecolor='white')
    print(f"\nVisualization saved to: {output_path}")
    plt.show()
    
    detector_correlations = [r['correlation'] for r in roads_data if r['has_detector']]
    
    if detector_correlations:
        fig, ax = plt.subplots(figsize=(10, 5))
        
        colors = [color_jam if c < 0 else color_no_jam for c in detector_correlations]
        ax.hist(detector_correlations, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Jam threshold (corr=0)')
        ax.set_xlabel('Flow-Occupancy Correlation', fontsize=12)
        ax.set_ylabel('Number of Roads', fontsize=12)
        ax.set_title(f'Distribution of Traffic Correlations in {CITY.replace("_", ", ")}\n(Negative = Traffic Jam)', fontsize=14)
        ax.legend()
        
        output_path2 = f"{CITY}_correlation_distribution.png"
        plt.savefig(output_path2, dpi=150, bbox_inches='tight')
        print(f"Correlation distribution saved to: {output_path2}")
        plt.show()

if __name__ == "__main__":
    main()
