import random
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import GAT
import os
from tqdm import tqdm

from prep_dataset import load_dataset

# ----------------------------
#region HYPERPARAMETERS
# ----------------------------
RAND_SEED = 42

# data specific:
NUM_CITIES = 2                  # set to None to use all cities
NUM_HIGHWAY_CLASSES = 7         # number of most common highway types to consider (rest will be "other" class)
VAL_RATIO = 0.05
TEST_RATIO = 0.1

# model specific:


#endregion
# ----------------------------


# ----------------------------
#region Random seed
# ----------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
#endregion
# ----------------------------
seed_everything(RAND_SEED)


# ----------------------------
#region  Load dataset
# ----------------------------
class FeatureBuilder(nn.Module):
    """
    Builds numerical and multi-hot encoded features for each node. The nodes are assigned the new attribute 'x' with these features:
    [ curvature, lanes, length, maxspeed, pop_density, is_highway_type_1, is_highway_type_2, ..., is_highway_type_N, is_highway_other ]
    """
    def __init__(self, num_highway_classes: int):
        super().__init__()
        all_highway_types = ["residential", "tertiary", "unclassified", "secondary", "primary", "trunk", "living_street", "motorway_link", "primary_link", "trunk_link", "motorway", "secondary_link", "tertiary_link", "busway", "crossing", "road", "escape", "yes", "via_ferrata", "alley", "emergency_bay"]
        self.highway_types = all_highway_types[:num_highway_classes]

    def forward(self, data):
        # numeric features (correlation NOT included, it's the target)
        num_feats = torch.stack(
            [
                data.curvature,
                data.lanes,
                data.length,
                data.maxspeed,
                data.pop_density,
            ],
            dim=-1,
        ).float()  # [N, 5]

        highway_lists = [ [v] if isinstance(v, str) else v for v in data.highway ]
        multi_hot_highway = torch.zeros((data.num_nodes, len(self.highway_types)+1), dtype=torch.float) # +1 for "other" class
        for node_idx, highway_list in enumerate(highway_lists):
            for h in highway_list:
                if h in self.highway_types:
                    h_idx = self.highway_types.index(h)
                else:
                    h_idx = len(self.highway_types)  # "other" class
                multi_hot_highway[node_idx, h_idx] = 1.0
        # multi_hot_highway: [N, num_highway_classes+1]

        data.x = torch.cat([num_feats, multi_hot_highway], dim=-1)  # [N, 5 + num_highway_classes+1]
        return data

def format_dataset(num_cities=None, num_highway_classes = 7):
    dataset = load_dataset(max_cities=num_cities)
    feature_builder = FeatureBuilder(num_highway_classes=num_highway_classes)
    for city, graph in dataset.items():
        dataset[city] = feature_builder(graph)
    return dataset
#endregion
# ----------------------------
dataset = format_dataset(num_cities=NUM_CITIES, num_highway_classes=NUM_HIGHWAY_CLASSES)


RESULTS_DIR = './results_v0'
# ----------------------------
#region Create results directory
# if os.path.exists(RESULTS_DIR):
#     raise FileExistsError(f"Results directory {RESULTS_DIR} already exists. Please remove it or choose a different name.")
os.makedirs(RESULTS_DIR, exist_ok=True)
#endregion
# ----------------------------

# ----------------------------
#region Train/Val/Test split
# ----------------------------
def add_masks(graph, city_name, save_dir, val_ratio=0.05, test_ratio=0.05):
    if os.path.exists(save_dir):
        mask = torch.load(save_dir)
    else:
        num_nodes = graph.num_nodes
        nodes_list = torch.arange(num_nodes)
        nodes_list = nodes_list[graph.correlation != 0]  # consider only nodes which have correlation != 0 (nodes with traffic data)

        # shuffle
        perm = torch.randperm(len(nodes_list))
        nodes_list = nodes_list[perm]

        num_val = int(len(nodes_list) * val_ratio)
        num_test = int(len(nodes_list) * test_ratio)

        val_nodes = nodes_list[:num_val]
        test_nodes = nodes_list[num_val:num_val + num_test]
        train_nodes = nodes_list[num_val + num_test:]

        # create mask ---> [num_nodes, 3] tensor where each row is (train, val, test) one-hot mask
        mask = torch.zeros((num_nodes, 3), dtype=torch.bool)
        mask[train_nodes, 0] = True
        mask[val_nodes, 1] = True
        mask[test_nodes, 2] = True

    graph.train_mask = mask[:, 0]
    graph.val_mask = mask[:, 1]
    graph.test_mask = mask[:, 2]

    torch.save(mask, save_dir)
    return graph

def train_val_test_split(dataset, val_ratio=0.05, test_ratio=0.05, verbose=True):
    splits = {}
    masks_dir = f"{RESULTS_DIR}/split_masks/"
    all_masks_exist = all([ os.path.exists(f"{masks_dir}/{city}.pt") for city in dataset ])

    if os.path.exists(masks_dir) and not all_masks_exist:
        raise FileExistsError(f"Some masks are missing in {masks_dir}. This probably happened because the program script has been ran with different number of cities.Please remove the directory or choose a different RESULTS_DIR for this run.")

    os.makedirs(masks_dir, exist_ok=True)

    if verbose:
        if not all_masks_exist: print("Creating train/val/test splits...")
        else: print("Loading existing train/val/test splits...")

    for city, graph in tqdm(dataset.items(), desc="Preparing train/val/test splits"):
        dataset[city] = add_masks(graph, city_name=city, save_dir=f"{masks_dir}/{city}.pt", val_ratio=val_ratio, test_ratio=test_ratio)

    if verbose:
        if not all_masks_exist: print(f"Train/val/test splits created and saved to {masks_dir}")
        else: print(f"Train/val/test splits loaded from {masks_dir}.")

    return dataset
#endregion
# ----------------------------
dataset = train_val_test_split(dataset, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO)

# ----------------------------
#region Model definition
# ----------------------------
class GATModel(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=64, num_layers=3, heads=4, dropout=0.2):
        super(GATModel, self).__init__()
        self.gat = GAT(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            v2=True,
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        out = self.gat(x, edge_index)
        return out
#endregion
# ----------------------------
model = GATModel(
    in_channels=dataset[next(iter(dataset))].x.size(-1),
    out_channels=1,
    hidden_channels=128,
    num_layers=3,
    heads=4,
    dropout=0.2,
)