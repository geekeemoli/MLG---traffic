# MLG---traffic: Predicting Critical Roads with Traffic Jams

## Project Documentation

**Authors:** Dario Vajda, Oliver Majer, Diego Bonaca  
**Institution:** University of Ljubljana  
**Course:** Machine Learning on Graphs (3rd Year, 1st Semester)  
**Medium Article:** ...

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Structure](#2-project-structure)
3. [Installation & Dependencies](#3-installation--dependencies)
4. [Data Sources](#4-data-sources)
5. [Module Documentation](#5-module-documentation)
   - [5.1 Source Code (`src/`)](#51-source-code-src)
   - [5.2 Data Processing (`data/`)](#52-data-processing-data)
   - [5.3 Traffic Analysis (`analyse_utd19/`)](#53-traffic-analysis-analyse_utd19)
   - [5.4 Model Training (`train/`)](#54-model-training-train)
6. [Pipeline Workflow](#6-pipeline-workflow)
7. [Data Reference](#7-data-reference)
8. [Configuration](#8-configuration)
9. [Usage Examples](#9-usage-examples)

---

## 1. Project Overview

This project aims to predict which road segments are critical (i.e., have high congestion probability) using graph-based machine learning. The model combines:

- **Graph topology** from OpenStreetMap road networks
- **Population density** from High Resolution Settlement Layer (HRSL) data
- **Traffic measurements** from the UTD19 dataset

### Key Features

- **Inductive Learning:** Train on some cities, test on unseen cities
- **Graph Neural Networks:** Uses Graph Attention Networks (GAT) for road segment classification
- **Binary Classification:** Critical vs non-critical road segments
- **Multi-city Support:** Processes 39 cities across 13 countries

---

## 2. Project Structure

```
MLG---traffic/
├── .gitignore                   # Git ignore rules
├── MLG_ Project Proposal.pdf    # Original project proposal
│
├── src/                         # Core source code
│   ├── popdensityV5.py          # Population density mapping
│   ├── popgraph_to_gpickle.py   # Graph generation with population data
│   └── prepare_final_graphs.py  # Final graph preparation with labels
│
├── data/                        # Data storage and processing
│   ├── README.md                # Data format documentation
│   ├── load_graph.py            # Graph loading utilities
│   ├── check_graphs.py          # Graph validation scripts
│   ├── merge_usa_population.py  # USA population data merger
│   ├── graphs/                  # Intermediate graphs with population
│   ├── final_graphs/            # Production-ready graphs
│   ├── population_data/         # Population CSV files by country
│   └── traffic_data/            # UTD19 detector data
│
├── analyse_utd19/               # Traffic jam detection
│   ├── README.md                # Explaining the data processing approach
│   ├── calculate_correlations.py # Correlation computation
│   ├── check_correlations.py    # Correlation analysis
│   ├── sample.py                # Data sampling utilities
│   └── all_correlations/        # Computed correlations
│
└── train/                       # Model training
    ├── train.py                 # Training script with GAT/MLP models
    ├── prep_dataset.py          # Dataset preparation
    └── results_*/               # Training results directories
```

---

## 3. Installation & Dependencies

### Required Python Packages

```python
# Core dependencies (install via pip)
networkx          # Graph manipulation
osmnx             # OpenStreetMap data retrieval
pandas            # Data manipulation
numpy             # Numerical computing
torch             # Deep learning framework
torch_geometric   # Graph neural networks
scipy             # Scientific computing (optional, for speedup)
xgboost           # XGBoost algorithm

# Visualization
matplotlib        # Plotting

# Data handling
shapely           # Geometric operations
tqdm              # Progress bars

# Standard library
gzip              # Compressed file handling
pickle            # Object serialization
csv               # CSV file parsing
math              # Mathematical functions
os                # Operating system interface
```

### Installation

```bash
pip install networkx osmnx pandas numpy torch torch_geometric scipy matplotlib shapely tqdm xgboost
```

## 4. Data Sources

### 4.1 City Networks
- **Source:** OpenStreetMap via `OSMnx` Python library
- **Format:** NetworkX MultiDiGraph → Line Graph
- **Network Type:** `"drive"` (road network for vehicles)

### 4.2 Population Density
- **Source:** High Resolution Settlement Layer (HRSL)
- **Format:** CSV with (Latitude, Longitude, Population) or (Longitude, Latitude, Population) tuples
- **Resolution:** 1 arc second × 1 arc second (~ 30m × 30m) per tile
- **Countries Supported:**
  - Austria (`aut`), Germany (`deu`), Switzerland (`che`), UK (`gbr`), France (`fra`), Italy (`ita`), Netherlands (`nld`), Spain (`esp`), USA (`usa`), Australia (`aus`), Taiwan (`twn`), Japan (`jpn`), Lithuania (`ltu`)

### 4.3 Traffic Data (UTD19)
- **Source:** ETH Zurich UTD19 Dataset
- **Files:**
  - `utd19_u.csv` - Traffic measurements (flow, occupancy)
  - `detectors_public.csv` - Detector coordinates
  - `links.csv` - Road network links
- **Measurements:** Flow (vehicles/time) and Occupancy (% of time detector occupied)

---

## 5. Module Documentation

### 5.1 Source Code (`src/`)

#### 5.1.1 `popdensityV5.py`

**Purpose:** Annotates graph nodes with population density values from HRSL CSV data.

**Main Function:**

```python
def get_density(G, csv_path, tile_half_ddeg=1.0/7200.0, assume_sorted_by_lat=False, 
                far_thresh_m=100.0, verbose=False) -> networkx.Graph:
    """
    Annotate graph nodes with 'pop_density' attribute.
    
    Parameters
    ----------
    G : networkx.Graph
        Line-graph where each node represents a road.
    csv_path : str
        Path to population CSV file.
    tile_half_ddeg : float
        Half tile size in decimal degrees (default: 1 arc-second).
    assume_sorted_by_lat : bool
        If True, enables early exit optimization for lat-sorted CSVs.
    far_thresh_m : float or None
        Maximum distance (meters) for road-tile assignment. None = no limit.
    verbose : bool
        If True, print statistics about tile/population coverage.
    
    Returns
    -------
    G : networkx.Graph
        Graph with 'pop_density' attribute added to each node.
    """
```

**Algorithm:**
1. Extract road centers from node geometry
2. Define bounding box from road coordinates
3. Parse CSV for population tiles within bounding box
4. **Pass 1:** Assign each road to its nearest tile and normalize by number of roads sharing that tile
5. **Pass 2:** Assign remaining tiles to nearest road

**Implementation Notes:**
- Uses `scipy.spatial.cKDTree` for fast nearest-neighbor queries (O(log n))
- Falls back to brute-force O(n²) search if scipy unavailable

---

#### 5.1.2 `popgraph_to_gpickle.py`

**Purpose:** Downloads city road networks from OpenStreetMap, creates line graphs, and enriches them with population density.

**City List:**
```python
cities = [
    "Augsburg, Germany", "Basel, Switzerland", "Berne, Switzerland",
    "Birmingham, UK", "Bolton, UK", "Bremen, Germany", "Bordeaux, France",
    "Cagliari, Italy", "Constance, Germany", "Darmstadt, Germany",
    "Essen, Germany", "Frankfurt, Germany", "Graz, Austria",
    "Groningen, Netherlands", "Hamburg, Germany", "Innsbruck, Austria",
    "Kassel, Germany", "London, UK", "Los Angeles, USA", "Lucerne, Switzerland",
    "Madrid, Spain", "Melbourne, Australia", "Manchester, UK", "Marseille, France",
    "Munich, Germany", "Paris, France", "Rotterdam, Netherlands", "Santander, Spain",
    "Speyer, Germany", "Strasbourg, France", "Stuttgart, Germany",
    "臺北市, Taiwan", "東京23区, Japan", "Torino, Italy", "Toulouse, France",
    "Utrecht, Netherlands", "Vilnius, Lithuania", "Wolfsburg, Germany", "Zurich, Switzerland"
]
# 臺北市 is Taipei, 東京23区 is Tokyo
```

**Key Functions:**

```python
def get_csv_path_for_city(city_str: str) -> str:
    """
    Get the population CSV path for a city string like "Graz, Austria".
    Maps country names to CSV file prefixes using COUNTRY_TO_CSV_PREFIX dict.
    """
```

**Pipeline:**
1. Download road network: `ox.graph_from_place(city, network_type="drive")`
2. Add geometry to edges without it
3. Create line graph: `nx.line_graph(G)`
4. Copy edge attributes to line graph nodes
5. Apply population density: `p.get_density(road_G, csv_path)`
6. Save as `.gpickle` file

**Output:** `data/graphs/{City_Country}_road_graph_with_popdensity.gpickle`

---

#### 5.1.3 `prepare_final_graphs.py`

**Purpose:** Creates production-ready graphs by:
1. Loading graphs with population density
2. Mapping traffic detectors to road segments
3. Computing and assigning correlation labels
4. Adding curvature features
5. Saving compressed final graphs

**Main Functions:**

```python
def load_correlation_data(correlation_file: str) -> pd.DataFrame:
    """
    Load detector correlations from analysis results.
    Uses only the correlation_ma column (moving average correlation).
    
    Returns
    -------
    DataFrame with columns: detector_id, correlation_ma
    """

def line_graph_to_osmnx_primal(G_roads) -> nx.MultiDiGraph:
    """
    Reconstruct a MultiDiGraph from a line graph.
    Used for osmnx.distance.nearest_edges() compatibility.
    """

def create_road_graphs_with_labels(detector_coords_file, correlation_file, cities) -> Dict:
    """
    Main pipeline for creating labeled graphs.
    
    Returns
    -------
    Dict[city_name, {graph, y_tensor, node_list, detector_mapping}]
    """

def calc_curvature(node_data: dict) -> float:
    """
    Calculate curvature for a road segment.
    
    Curvature = (path_length / chord_length) - 1
    
    - Straight roads have curvature ≈ 0
    - Curved roads have higher values
    - Clamped to [0, 5] and log-scaled
    """

def save_graphs_and_labels(results: Dict, output_dir: str):
    """
    Save processed graphs as compressed pickle files.
    
    Final node attributes kept:
    - highway, length, lanes, maxspeed
    - pop_density, correlation, curvature
    """
```

**Output Format:** `data/final_graphs/{City_Country}_graph.pkl.gz`

---

### 5.2 Data Processing (`data/`)

#### 5.2.1 `load_graph.py`

**Purpose:** Utility functions for loading saved graphs.

```python
def load_graph_nx(path: str) -> networkx.Graph:
    """Returns a Networkx graph of a city at the given path."""
    
def load_graph_pyg(path: str) -> torch_geometric.data.Data:
    """Returns a PyG graph of a city at the given path."""
```

---

#### 5.2.2 `check_graphs.py`

**Purpose:** Inspection and statistics aggregation for processed graphs.

**Key Functions:**

```python
def add_curvature(node_data: dict) -> dict:
    """
    Calculate curvature for a node based on its geometry. The curvature is defined as the ratio of the path length
    along the polyline to the straight-line distance between the endpoints, normalized so that straight lines have curvature 0.
    The curvature is clamped to [0, 5] and then log-scaled.

    Args:
        node_data (dict): Node data containing a 'geometry' key with a shapely LineString.
    Returns:
        dict: Node data with an added 'curvature' key.
    """

def main(path: str) -> dict:
    """
    Analyze a graph file and return statistics.
    """
```

---

#### 5.2.3 `merge_usa_population.py`

**Purpose:** Extracts and consolidates Los Angeles population data from multiple USA CSV files.

Unlike other countries (which have a single CSV file each), the USA population data is split across 6 separate CSV files. Since the data is not geographically sorted, Los Angeles entries appear in all 6 files. This script filters and merges the relevant rows into a single file for efficient processing.

**Bounding Box:**
- Latitude: 33° to 35° N
- Longitude: 119° to 117° W

**Output:** `population_data/population_usa_los_angeles.csv`

---

### 5.3 Traffic Analysis (`analyse_utd19/`)

#### 5.3.1 `calculate_correlations.py`

**Purpose:** Detects traffic jams by computing correlation between flow and occupancy.

**Key Insight:**
- High correlation (≈1): Normal traffic flow proportional to occupancy
- Low/negative correlation: Traffic jam (high occupancy but low flow)

**Main functions:**

```python
def compute_correlation(df) -> Tuple[str, str]:
    """
    Compute correlation between flow and occupancy
    at data points where occupancy > 66th percentile,
    and also for their moving averages.
    """

def plot_flow_occ_over_time_with_ma(data, out_path, N, plot):
    """
    Plot 'flow', 'occ', and their ratio over time (stacked subplots)
    from a list of dict records and save to a file.

    In each subplot, plot both the original values and
    an N-point moving average.

    Parameters
    ----------
    data : list of dict
        Each element must have keys: 'day', 'interval', 'flow', 'occ'
    out_path : str
        File path to save the generated plot (e.g. 'plot.png')
    N : int
        Window size for the moving average (in number of points).
    """
```

**Command Line Usage:**
```bash
python3 calculate_correlations.py \
 --should_plot=True --cities=madrid \
 --sampled_path=utd_samples/sampled_madrid.csv \
 --output_dir=madrid_correlations
```

---

#### 5.3.2 `check_correlations.py`

**Purpose:** Analyzes computed correlations and generates histograms.

**Features:**
- Loads correlations from `sampled_correlations.csv`
- Counts NaN values and negative correlations
- Generates histogram plots

**Output:** `correlations_plot.png`

---

#### 5.3.3 `sample.py`

**Purpose:** Samples UTD19 data by city.

```python
def sample_utd_by_city(cities, utd19_path, sampled_path):
    """
    Sample UTD data for specified cities and save to 'sampled_utd19.csv'.
    Args:
        cities (list): List of city names to filter data by. If None, all data is included.
        utd19_path (str): Path to the input UTD CSV file.
        sampled_path (str): Path to the output sampled CSV file.
    """
```

---

### 5.4 Model Training (`train/`)

#### 5.4.1 `prep_dataset.py`

**Purpose:** Loads and prepares graph dataset for training.

```python
def load_graph_nx(path: str) -> networkx.Graph:
    """Returns a Networkx graph of a city at the given path."""

def load_graph_pyg(path: str) -> torch_geometric.data.Data:
    """Returns a PyG graph of a city at the given path."""

def load_dataset(max_cities: int = None) -> Dict[str, Data]:
    """
    Load city graphs from final_graphs directory.
    
    Parameters
    ----------
    max_cities : int, optional
        Limit number of cities loaded
    
    Returns
    -------
    Dict mapping city names to PyG Data objects
    
```

---

#### 5.4.2 `train.py`

**Purpose:** Main training script with GAT ensemble and MLP baseline models.

**Hyperparameters:**

```python
RAND_SEED = 42
NUM_HIGHWAY_CLASSES = 7      # Top highway types (rest → "other")
VAL_RATIO = 0.1
TEST_RATIO = 0.0

MODEL = "mlp_baseline"    
N_MODELS = 1                 # number of GAT models in the ensemble
HIDDEN_DIM = 64
NUM_LAYERS = 3
NUM_HEADS = 4                # For GAT attention
DROPOUT = 0

LEARNING_RATE = 1e-2
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 500
LOSS_WEIGHT = 2.5            # Positive class weight
```

**Model Architectures:**

```python
class GATModelEnsemble(nn.Module):
    """
    Ensemble of Graph Attention Networks.
    
    Uses PyG's GAT v2 implementation with:
    - Multi-head attention
    - Configurable depth and width
    - Dropout regularization
    
    Forward: Averages predictions from all ensemble members
    """

class MLPBaselineModel(nn.Module):
    """
    Simple MLP baseline (no graph structure).
    
    Architecture:
    - Linear → ReLU → Dropout (repeated)
    - Final Linear → output
    """
```

**Feature Builder:**

```python
class FeatureBuilder(nn.Module):
    """
    Builds numerical and multi-hot encoded features for each node. The nodes are assigned the new attribute 'x' with these features:
    [ curvature, lanes, length, maxspeed, pop_density, is_highway_type_1, is_highway_type_2, ..., is_highway_type_N, is_highway_other ]
    """
```

**Loss Function:**

```python
def loss_fn(logits, target, split_mask=None, epsilon=0.02, return_stats=False):
    """
    Weighted binary cross-entropy loss.
    
    Target encoding:
    - Jam (correlation < 0) → positive class (y=1)
    - No jam (correlation > 0) → negative class (y=0)
    - Unknown (correlation = 0) → excluded from loss
    
    Weighting:
    1. Confidence weight: |correlation| + epsilon
    2. Class weight: LOSS_WEIGHT for jams, 1/LOSS_WEIGHT for no-jams
    """
```

**Training Loop:**
1. Load and format dataset (add `x` features)
2. Create train/val/test masks (by node, within each city)
3. For each epoch:
   - Accumulate gradients across all cities
   - Single optimizer step
   - Log train/val loss and confusion matrix stats

**Output:**
- `{RESULTS_DIR}/model.pth` - Saved model weights
- `{RESULTS_DIR}/training_log.txt` - Training logs
- `{RESULTS_DIR}/hyperparameters.json` - Configuration
- `{RESULTS_DIR}/split_masks/` - Train/val/test masks

---

## 6. Pipeline Workflow

### Complete Data Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA COLLECTION                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  OpenStreetMap ──► OSMnx ──► Road Network Graph                 │
│                                                                 │
│  HRSL ──► CSV ──► Population Data                               │
│                                                                 │
│  UTD19 Dataset ──► CSV ──► Traffic Measurements                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PREPROCESSING                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. popgraph_to_gpickle.py:                                     │
│     Road Graph + Population ──► {city}_with_popdensity.gpickle  │
│                                                                 │
│  2. calculate_correlations.py:                                  │
│     Traffic Data ──► correlations.csv                           │
│                                                                 │
│  3. prepare_final_graphs.py:                                    │
│     All Sources ──► {city}_graph.pkl.gz (labeled)               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL TRAINING                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  prep_dataset.py:                                               │
│     Load .pkl.gz ──► PyG Data objects                           │
│                                                                 │
│  train.py:                                                      │
│     Build features ──► Train GAT/MLP ──► Save model.pth         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Data Reference

### Graph Node Attributes

| Attribute | Type | Description | Range |
|-----------|------|-------------|-------|
| `highway` | str/List[str] | Road type classification | See highway types table |
| `length` | float | Road segment length | meters |
| `lanes` | float | Average number of lanes | ≥1.0 |
| `maxspeed` | float | Speed limit | km/h (default: 50) |
| `pop_density` | float | Population near road | people |
| `correlation` | float | Flow-occupancy correlation | [-1, 1], 0=unknown |
| `curvature` | float | Road curvature measure | [0, log(6)] |

### Highway Types (by frequency)

| Type | Description |
|------|-------------|
| residential | Local residential streets |
| tertiary | Local connecting roads |
| unclassified | Minor roads |
| secondary | Regional roads |
| primary | Major roads |
| trunk | High-importance roads |
| living_street | Pedestrian priority zones |
| motorway_link | Highway on/off ramps |
| motorway | Highways/freeways |

---

## 8. Configuration

### Environment Variables

The project uses relative paths from script locations. Key directories:

- `SCRIPT_DIR` - Directory containing the running script
- `POP_DATA_DIR` - `../data/population_data/` relative to `src/`

### Country to CSV Prefix Mapping

```python
COUNTRY_TO_CSV_PREFIX = {
    "Austria": "aut", "Germany": "deu", "Switzerland": "che",
    "UK": "gbr", "France": "fra", "Italy": "ita",
    "Netherlands": "nld", "Spain": "esp", "USA": "usa",
    "Australia": "aus", "Taiwan": "twn", "Japan": "jpn",
    "Lithuania": "ltu"
}
```

### Training Configuration

Edit constants at top of `train/train.py`:

```python
# Data
NUM_CITIES = None           # None = all cities
NUM_HIGHWAY_CLASSES = 7     # Top k highway types

# Model
MODEL = "gat_ensemble"      # or "mlp_baseline"
HIDDEN_DIM = 64
NUM_LAYERS = 3

# Training
NUM_EPOCHS = 500
LEARNING_RATE = 1e-2
LOSS_WEIGHT = 2.5           # Jam class weight
```

---

## 9. Usage Examples

### Generate Graphs for New City

```python
import osmnx as ox
import networkx as nx
from shapely.geometry import LineString
import src.popdensityV5 as p

# 1. Download road network
city = "Vienna, Austria"
G = ox.graph_from_place(city, network_type="drive")

# 2. Add geometry to all edges
for u, v, k, data in G.edges(keys=True, data=True):
    if 'geometry' not in data:
        x1, y1 = G.nodes[u]['x'], G.nodes[u]['y']
        x2, y2 = G.nodes[v]['x'], G.nodes[v]['y']
        data['geometry'] = LineString([(x1, y1), (x2, y2)])

# 3. Create line graph (roads as nodes)
road_G = nx.line_graph(G)
for u, v, k, data in G.edges(keys=True, data=True):
    if (u, v, k) in road_G:
        road_G.nodes[(u, v, k)].update(data)

# 4. Add population density
csv_path = "data/population_data/aut_general_2020.csv"
road_G = p.get_density(road_G, csv_path, verbose=True)

# 5. Save graph
import pickle
with open("vienna_graph.gpickle", "wb") as f:
    pickle.dump(road_G, f)
```

### Compute Correlations for New City

```bash
# 1. Sample UTD data for the city
python analyse_utd19/calculate_correlations.py \
    --cities=vienna \
    --sampled_path=utd_samples/sampled_vienna.csv \
    --output_dir=vienna_correlations \
    --should_plot=True
```

### Load and Inspect a Graph

```python
from data.load_graph import load_graph_pyg

graph = load_graph_pyg("data/final_graphs/Graz_Austria_graph.pkl.gz")
print(graph)

# Count traffic-labeled nodes
labeled = (graph.correlation != 0).sum().item()
print(f"Traffic-labeled nodes: {labeled}")
```

### Train a Model

```bash
cd train
python train.py
```

Results saved to `RESULTS_DIR` (default: `./result_mlp_baseline_v1/`).

---

## Appendix: City Statistics

| City | Nodes | Edges | Labeled Nodes |
|------|------:|------:|--------------:|
| Rotterdam, Netherlands | 25,579 | 63,466 | 112 |
| Melbourne, Australia | 499,153 | 1,232,527 | 263 |
| Graz, Austria | 11,256 | 30,845 | 152 |
| Munich, Germany | 36,326 | 103,316 | 276 |
| Hamburg, Germany | 52,871 | 140,485 | 206 |
| Paris, France | 18,178 | 36,811 | 186 |
| London, UK | 303,696 | 812,652 | 3,532 |
| Los Angeles, USA | 136,122 | 423,329 | 1,480 |
| Tokyo, Japan | 305,402 | 880,439 | 188 |
| Zurich, Switzerland | 10,571 | 27,088 | 612 |
| **Total** | ~2M | ~5M | **12,474** |

---
