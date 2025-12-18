import random
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import GAT
import os
from tqdm import tqdm
import json

from prep_dataset import load_dataset

# --------------------------------------------------------
#region HYPERPARAMETERS
# --------------------------------------------------------
RAND_SEED = 42

# data specific:
NUM_CITIES = None               # set to None to use all cities
NUM_HIGHWAY_CLASSES = 7         # number of most common highway types to consider (rest will be "other" class)
VAL_RATIO = 0.1
TEST_RATIO = 0.0

# model specific:
MODEL="mlp_baseline"
N_MODELS = 1                    # number of GAT models in the ensemble
HIDDEN_DIM = 64
NUM_LAYERS = 3
NUM_HEADS = 4
DROPOUT = 0

# training specific:
RESULTS_DIR = './result_mlp_baseline_v1'
LEARNING_RATE = 1e-2
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 500
LOSS_WEIGHT = 2.5 # weight for the positive class (traffic jam) in the loss function
MODEL_SAVE_PATH = f"{RESULTS_DIR}/model.pth"
LOGS_FILE = f"{RESULTS_DIR}/training_log.txt"

hyperparameters = {
    "RAND_SEED": RAND_SEED,
    "NUM_CITIES": NUM_CITIES,
    "NUM_HIGHWAY_CLASSES": NUM_HIGHWAY_CLASSES,
    "VAL_RATIO": VAL_RATIO,
    "TEST_RATIO": TEST_RATIO,
    "MODEL": MODEL,
    "N_MODELS": N_MODELS,
    "HIDDEN_DIM": HIDDEN_DIM,
    "NUM_LAYERS": NUM_LAYERS,
    "NUM_HEADS": NUM_HEADS,
    "DROPOUT": DROPOUT,
    "RESULTS_DIR": RESULTS_DIR,
    "LEARNING_RATE": LEARNING_RATE,
    "WEIGHT_DECAY": WEIGHT_DECAY,
    "NUM_EPOCHS": NUM_EPOCHS,
    "MODEL_SAVE_PATH": MODEL_SAVE_PATH,
    "LOGS_FILE": LOGS_FILE,
}
#endregion
# --------------------------------------------------------

# --------------------------------------------------------
#region Create results directory and save hyperparameters
# if os.path.exists(RESULTS_DIR):
#     raise FileExistsError(f"Results directory {RESULTS_DIR} already exists. Please remove it or choose a different name.")
os.makedirs(RESULTS_DIR, exist_ok=True)
with open(f"{RESULTS_DIR}/hyperparameters.json", "w") as f:
    json.dump(hyperparameters, f, indent=4)
#endregion
# --------------------------------------------------------

# --------------------------------------------------------
#region Random seed
# --------------------------------------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
seed_everything(RAND_SEED)
#endregion
# --------------------------------------------------------

# --------------------------------------------------------
#region  Load dataset
# --------------------------------------------------------
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
# --------------------------------------------------------
dataset = format_dataset(num_cities=NUM_CITIES, num_highway_classes=NUM_HIGHWAY_CLASSES) # <--- adds 'x' attribute to each graph with numerical node features

# --------------------------------------------------------
#region Train/Val/Test split - train_val_test_split(dataset, val_ratio, test_ratio)
# --------------------------------------------------------
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
# --------------------------------------------------------
dataset = train_val_test_split(dataset, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO) # <--- adds train_mask, val_mask, test_mask to each graph

# --------------------------------------------------------
#region GAT Model definition - create_gat_model(dataset)
# --------------------------------------------------------
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

def create_gat_model(dataset):
    model = GATModelEnsemble(
        n_models=N_MODELS,
        in_channels=dataset[next(iter(dataset))].x.size(-1),
        out_channels=1,
        hidden_channels=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        heads=NUM_HEADS,
        dropout=DROPOUT,
    )

    # print number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created with {num_params} trainable parameters.")

    return model
#endregion
# --------------------------------------------------------

# --------------------------------------------------------
#region MLP Baseline Model definition - create_baseline_model(dataset)
# --------------------------------------------------------
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

def create_mlp_baseline_model(dataset):
    model = MLPBaselineModel(
        in_channels=dataset[next(iter(dataset))].x.size(-1),
        hidden_channels=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    )

    # print number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Baseline model created with {num_params} trainable parameters.")

    return model
#endregion
# --------------------------------------------------------

# --------------------------------------------------------
#region Training setup - setup_training(dataset), loss_fn(...), eval_metric(...)
# --------------------------------------------------------
def setup_training(dataset, create_model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = create_model(dataset)
    model = model.to(device)

    for city in dataset:
        dataset[city] = dataset[city].to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    return dataset, model, optimizer, device

#region old loss fn
# def loss_fn(logits, target, split_mask=None, epsilon=0.02, return_stats=False):
#     """
#     Cross entropy loss where sigmoid(logits) is compared with (target+1)/2 and at each position the loss is weighted by abs(target)+epsilon, but only for nodes where target is not zero.
#     - logits: [ num_nodes ] output logits tensor
#     - target: [ num_nodes ] tensor with true correlation values in the range [-1, 1] (0 for nodes without traffic data)
#     - mask: [ num_nodes, ]

#     Returns: scalar loss value
#     """
#     mask = (target != 0) & split_mask if split_mask is not None else (target != 0)
#     logits = logits[mask]
#     target = target[mask]

#     confidence_weights = torch.abs(target) + epsilon
#     confidence_weights = confidence_weights / confidence_weights.mean()

#     traffic_jam_count = (target < 0).sum().item()
#     traffic_jam_weight = min(1 - (traffic_jam_count / len(target)), 0.9)

#     class_weights = torch.where(target >= 0, 1.0 - traffic_jam_weight, traffic_jam_weight)

#     weights = confidence_weights * class_weights
#     target_scaled = (target + 1) / 2
#     loss = F.binary_cross_entropy_with_logits(logits, target_scaled, weight=weights, reduction='mean')

#     if return_stats:
#         with torch.no_grad():
#             # (inverting positive and negative classes for better interpretability)
#             preds = (torch.sigmoid(logits) <= 0.5).float()
#             targets_binary = (target_scaled <= 0.5).float()
#             stats = {
#                 "tp": ((preds == 1) & (targets_binary == 1)).sum().item(),
#                 "tn": ((preds == 0) & (targets_binary == 0)).sum().item(),
#                 "fp": ((preds == 1) & (targets_binary == 0)).sum().item(),
#                 "fn": ((preds == 0) & (targets_binary == 1)).sum().item(),
#             }
#         return loss, stats

#     return loss
#endregion

def loss_fn(logits, target, split_mask=None, epsilon=0.02, return_stats=False):
    mask = (target != 0)
    if split_mask is not None:
        mask = mask & split_mask

    logits = logits[mask]
    target = target[mask]

    # jam is the positive class
    y = (target < 0).float()  # -1 -> 1 (jam), +1 -> 0 (no jam)

    # confidence weights (optional)
    conf = target.abs() + epsilon
    conf = conf / conf.mean().clamp_min(1e-12)

    # class weights: inverse frequency (upweight minority)
    pos = y.sum()
    neg = (1 - y).sum()
    if (pos == 0 or neg == 0) and False:
        # Only one class present: skip class-balancing for this graph/split
        cls_w = torch.ones_like(y)
    else:
        # w_pos = min((neg / pos).detach(), 10)
        # w_neg = min((pos / neg).detach(), 10)
        w_pos = LOSS_WEIGHT
        w_neg = 1/LOSS_WEIGHT
        cls_w = torch.where(y == 1, w_pos, w_neg)

    weights = conf * cls_w

    per_elem = F.binary_cross_entropy_with_logits(logits, y, weight=weights, reduction="none")
    loss = per_elem.sum() / weights.sum().clamp_min(1e-12)

    if return_stats:
        with torch.no_grad():
            preds = (torch.sigmoid(logits) >= 0.5).float()
            stats = {
                "tp": ((preds == 1) & (y == 1)).sum().item(),
                "tn": ((preds == 0) & (y == 0)).sum().item(),
                "fp": ((preds == 1) & (y == 0)).sum().item(),
                "fn": ((preds == 0) & (y == 1)).sum().item(),
            }
        return loss, stats

    return loss

def eval_metric(stats, method="accuracy"):
    if method == "accuracy":
        correct = stats["tp"] + stats["tn"]
        total = stats["tp"] + stats["tn"] + stats["fp"] + stats["fn"]
        return correct / total if total > 0 else 0.0
    elif method == "precision":
        return stats["tp"] / (stats["tp"] + stats["fp"]) if (stats["tp"] + stats["fp"]) > 0 else 0.0
    elif method == "recall":
        return stats["tp"] / (stats["tp"] + stats["fn"]) if (stats["tp"] + stats["fn"]) > 0 else 0.0
    elif method == "f1":
        precision = eval_metric(stats, method="precision")
        recall = eval_metric(stats, method="recall")
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    else:
        raise ValueError(f"Unknown evaluation method: {method}")

def print_dataset_stats(dataset):
    pos_total, neg_total = 0, 0
    print(f"{'City:':<25} | {'Positive samples (jams)':<30} | {'Negative samples (no jams)':<30}")
    print("-" * 90)
    for city, graph in dataset.items():
        mask = graph.train_mask | graph.val_mask | graph.test_mask
        target = graph.correlation[mask]
        pos = (target < 0).sum().item()
        neg = (target > 0).sum().item()
        pos_total += pos
        neg_total += neg
        print(f"{city:<25} | {pos:<30} | {neg:<30}")
    print(f"Overall dataset - Positive samples (jams): {pos_total}, Negative samples (no jams): {neg_total}")
#endregion
# --------------------------------------------------------

# --------------------------------------------------------
#region Training loop - train(dataset, log_file=None)
# --------------------------------------------------------
def train(dataset, create_model, log_file=None):
    dataset, model, optimizer, device = setup_training(dataset, create_model)

    print_dataset_stats(dataset)

    num_cities = len(dataset.keys())

    if log_file is not None:
        log_f = open(log_file, "w")
    else:
        log_f = None

    for epoch in tqdm(range(1, NUM_EPOCHS + 1)):
        optimizer.zero_grad()
        
        total_train_loss, total_val_loss = 0, 0
        summarised_train_stats, summarised_val_stats = {"tp":0, "tn":0, "fp":0, "fn":0}, {"tp":0, "tn":0, "fp":0, "fn":0}
        for city, graph in dataset.items():

            model.train()
            out = model(graph).squeeze()
            train_loss, train_stats = loss_fn(out, graph.correlation, split_mask=graph.train_mask, return_stats=True)

            model.eval()
            with torch.no_grad():
                val_out = model(graph).squeeze()
                val_loss, val_stats = loss_fn(val_out, graph.correlation, split_mask=graph.val_mask, return_stats=True)

            total_train_loss += train_loss
            total_val_loss += val_loss.item()
            summarised_train_stats = { k: summarised_train_stats[k] + train_stats[k] for k in summarised_train_stats }
            summarised_val_stats = { k: summarised_val_stats[k] + val_stats[k] for k in summarised_val_stats }

        total_train_loss.backward()
        optimizer.step()

        avg_train_loss = total_train_loss.item() / num_cities
        avg_val_loss = total_val_loss / num_cities
        print(
            f"Epoch {epoch:03d}: Train Loss: {avg_train_loss:.4f} (TP, FP, TN, FN)=({summarised_train_stats['tp']}, {summarised_train_stats['fp']}, {summarised_train_stats['tn']}, {summarised_train_stats['fn']}), Val Loss: {avg_val_loss:.4f} (TP, FP, TN, FN)=({summarised_val_stats['tp']}, {summarised_val_stats['fp']}, {summarised_val_stats['tn']}, {summarised_val_stats['fn']})",
            file=log_f
        )
    
    if log_f is not None:
        log_f.close()
    
    return model

#endregion
# --------------------------------------------------------
create_model = create_gat_model if MODEL == "gat_ensemble" else create_mlp_baseline_model
trained_model = train(dataset, create_model=create_model, log_file=LOGS_FILE)

# --------------------------------------------------------
#region Save the trained model
# --------------------------------------------------------
def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
#endregion
# --------------------------------------------------------
save_model(trained_model, MODEL_SAVE_PATH)