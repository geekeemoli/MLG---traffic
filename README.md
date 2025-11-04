# Predicting Critical Roads with Traffic Jams

Authors: Dario Vajda, Oliver Majer, Diego Bonaca

University of Ljubljana

This repository contains materials for the project "Predicting Critical Roads with Traffic Jams". The goal is to train a model that identifies road segments most likely to cause traffic jams in large urban road networks. The model will combine graph topology, population density and traffic measurements and will be evaluated in an inductive setting (train on some cities, test on unseen cities).

## 1. Introduction

We aim to predict which road segments are critical (i.e., have high congestion probability) using graph-based machine learning. A reliable model could be used in planning and to detect potential issues before physical changes are made to the road network.

## 2. Data

2.1 City networks
- Source: OpenStreetMap (OSM) via the `OSMnx` Python library. OSMnx will be used to download and convert street networks into graph representations suitable for GNNs.

2.2 Population density
- Source: High Resolution Settlement Layer (HRSL) published as GeoTIFF rasters (∼30m×30m resolution). We will extract population estimates per spatial cell and map those to nearby road segments or nodes.

2.3 Traffic data
- Source: UTD19 dataset (ETH). UTD19 provides traffic flow information from thousands of stationary detectors and will serve as the ground truth for model training and evaluation.

Notes on preprocessing
- Convert HRSL GeoTIFFs to point or CSV formats (latitude, longitude, population) or resample onto the road graph. Use `rasterio` or GDAL for raster handling. Align spatial reference systems when joining population to graph nodes.

## 3. Model

Problem framing
- Binary classification on road segments: critical vs non-critical.
- Inductive evaluation: train on a set of cities and test on cities not seen during training.

Model family choices considered
- Graph Transformers — global self-attention can capture long-range dependencies but scale O(n^2) in memory and are impractical for very large city graphs.
- Message-Passing GNNs (MP-GNNs) — scale close to O(n) on sparse graphs and are more practical for large road networks.

Chosen approach
- We will use a message-passing GNN with local attention: Graph Attention Network (GAT). GAT combines expressivity (learnable attention weights over neighbors) with reasonable scaling on sparse road networks.

Implementation notes
- Node / edge features can include: topology features (degree, betweenness), sensor time-aggregates (average flow, peak counts), mapped population around the node/segment, road type and length.
- If temporal dynamics are important, extend GAT with temporal encoding (e.g., additional temporal features, temporal convolutions, or a temporal GNN variant).

## 4. Metrics and evaluation

Primary metrics
- Precision, recall and the F_2-score (β=2), where recall is emphasized because false negatives (missing a critical road) are costlier than false positives.

Secondary metrics
- ROC AUC, Recall@k%, Brier score, and Expected Calibration Error (ECE) for calibration analysis.

F_β formula
$$
F_{\beta} = (1 + \beta^2) \cdot \frac{P \cdot R}{\beta^2 P + R}
$$

Data splits & protocol
- Inductive split by city: 70% cities for training, 10% for validation and 20% for testing. Cities used for testing must be unseen during training and validation.
- Use validation to tune hyperparameters and select the classification threshold that maximizes F_2.

## 5. Experimental setup

- Train / val / test splits are per-city (not per-segment) to test generalization across urban networks.
- Use class weighting or oversampling if the critical class is rare.
- Provide per-city and aggregated metrics in results.

## 6. Implementation & dependencies

Languages and libraries (suggested)
- Python 3.8+
- PyTorch for model implementation
- PyTorch Geometric (PyG) or DGL for graph operations
- OSMnx for street network extraction
- rasterio / GDAL for HRSL raster processing
- pandas, numpy, scikit-learn for standard data handling and metrics

Example packages to include in `requirements.txt` (pin versions as needed)
- torch
- torch-geometric (or dgl)
- osmnx
- rasterio
- pandas
- numpy
- scikit-learn

## 7. Repository structure (recommended)

- `data/` — raw downloads and processed datasets (do not commit large raw files; provide download scripts)
- `src/` — code: `data/` (preprocessing), `models/` (GAT and helpers), `train.py`, `evaluate.py`
- `notebooks/` — EDA, preprocessing demonstrations and visualizations
- `configs/` — experiment configuration files (YAML)
- `experiments/` — saved checkpoints, logs, and evaluation outputs (gitignored large files)
- `README.md` — this file

## 8. Getting started (quick)

1) Create and activate a Python virtual environment (PowerShell):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

2) Download data (example placeholder script):

```powershell
python src/data/download_osm.py --city "Ljubljana, Slovenia" --output data/raw/osm
python src/data/download_hrsl.py --region "Slovenia" --output data/raw/hrsl
# UTD19 requires following dataset-specific download instructions
```

3) Preprocess and build graphs:

```powershell
python src/data/preprocess.py --osm data/raw/osm --hrsl data/raw/hrsl --traffic data/raw/utd19 --out_dir data/processed
```

4) Train (example):

```powershell
python src/train.py --config configs/gat_experiment.yaml
```

5) Evaluate:

```powershell
python src/evaluate.py --checkpoint experiments/checkpoint.pt --data_dir data/processed
```

Replace these example scripts with the actual script names when they exist in `src/`.

## 9. Tips & edge cases

- Handle missing traffic sensor data with masking or imputation.
- Use graph sampling or localized minibatches for very large graphs (GraphSAGE-style neighborhoods, Cluster-GCN).
- If population must be aggregated, try multiple radii around nodes to find the best mapping.

## 10. Next steps (recommended)

1. Add an exact `requirements.txt` with pinned package versions.
2. Implement example scripts: `src/data/preprocess.py`, `src/train.py`, `src/evaluate.py` and simple configs in `configs/`.
3. Provide a small sample dataset (synthetic or a subset) under `data/sample/` for quick demos and CI tests.
4. Add a `LICENSE` file and contributor guidelines.

## 11. Contact & license

Project authors: Dario Vajda, Oliver Majer, Diego Bonaca

University of Ljubljana

License: (add license file, e.g., MIT or Apache-2.0)

---

If you'd like, I can now:
- commit a `requirements.txt` with suggested packages,
- add minimal skeleton scripts for `preprocess.py`, `train.py` and `evaluate.py`,
- or create a small synthetic sample dataset for a runnable example. Which would you prefer next?