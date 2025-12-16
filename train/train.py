import random
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import GAT

from prep_dataset import load_dataset

# ----------------------------
# Reproducibility
# ----------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ----------------------------
# Build node features X and labels
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

dataset = format_dataset(num_cities=1, num_highway_classes=7)