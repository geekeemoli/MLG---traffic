"""
Population density mapper for road graphs.

Two-pass assignment algorithm:
  1. Each road -> nearest tile (within threshold)
  2. Remaining tiles -> nearest road (within threshold)

This ensures maximum coverage of population data.

Usage:
    from src.popdensity import get_density
    G = get_density(G, csv_path, far_thresh_m=100.0)
"""

import csv
import math
from collections import defaultdict

try:
    from scipy.spatial import cKDTree
    import numpy as np
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def get_density(G, csv_path, tile_half_ddeg=1.0/7200.0, assume_sorted_by_lat=True, far_thresh_m=100.0, verbose=False):
    """
    Annotate graph nodes with 'pop_density' attribute.

    Parameters
    ----------
    G : networkx.Graph
        Line-graph where each node represents a road.
    csv_path : str
        Path to population CSV (columns: lon, lat, population).
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

    def _center_from_node_data(data):
        geom = data.get('geometry')
        if geom is not None:
            try:
                c = geom.centroid
                return float(c.x), float(c.y)
            except Exception:
                pass
        x = data.get('x') or data.get('lon') or data.get('long')
        y = data.get('y') or data.get('lat')
        if x is not None and y is not None:
            return float(x), float(y)
        return None

    # Check if already annotated
    if any('pop_density' in d for _, d in G.nodes(data=True)):
        return G

    # Build road centers list
    centers = []
    index_to_node = []

    for n, data in G.nodes(data=True):
        data['pop_density'] = 0.0
        center = _center_from_node_data(data)
        if center is not None:
            centers.append(center)
            index_to_node.append(n)

    if not centers:
        return G

    # Compute bounding box
    lons = [c[0] for c in centers]
    lats = [c[1] for c in centers]
    west = min(lons) - tile_half_ddeg
    east = max(lons) + tile_half_ddeg
    south = min(lats) - tile_half_ddeg
    north = max(lats) + tile_half_ddeg

    # Read CSV tiles inside bbox
    tile_pop = {}
    unique_csv_centers = {}

    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        lon_field, lat_field, pop_field = reader.fieldnames[:3]
        seen_in_range = False

        for row in reader:
            try:
                lon = float(row[lon_field])
                lat = float(row[lat_field])
                val = float(row[pop_field])
            except Exception:
                continue

            if lon < west or lon > east or lat < south or lat > north:
                if assume_sorted_by_lat:
                    if lat < south:
                        continue
                    if lat > north and seen_in_range:
                        break
                continue

            seen_in_range = True
            key = (row[lon_field].strip(), row[lat_field].strip())
            tile_pop[key] = tile_pop.get(key, 0.0) + val
            if key not in unique_csv_centers:
                unique_csv_centers[key] = (lon, lat)

    if not tile_pop:
        return G

    # Convert threshold from meters to approximate degrees (at this latitude)
    # 1 degree latitude ≈ 111km, 1 degree longitude ≈ 111km * cos(lat)
    avg_lat = (south + north) / 2
    deg_per_m = 1.0 / 111000.0  # approximate
    far_thresh_deg = far_thresh_m * deg_per_m if far_thresh_m else None

    # Total population in bbox for coverage calculation
    total_pop_bbox = sum(tile_pop.values())

    if HAS_SCIPY:
        return _get_density_fast(G, centers, index_to_node, tile_pop, unique_csv_centers, far_thresh_deg, avg_lat, verbose, total_pop_bbox)
    else:
        return _get_density_slow(G, centers, index_to_node, tile_pop, unique_csv_centers, far_thresh_deg, verbose, total_pop_bbox)


def _get_density_fast(G, centers, index_to_node, tile_pop, unique_csv_centers, far_thresh_deg, avg_lat, verbose, total_pop_bbox):
    """Fast version using scipy KD-Tree."""
    
    # Build arrays
    road_coords = np.array(centers)
    tile_keys = list(unique_csv_centers.keys())
    tile_coords = np.array([unique_csv_centers[k] for k in tile_keys])
    tile_pops = np.array([tile_pop[k] for k in tile_keys])
    
    # Scale longitude to account for latitude (approximate equal-area)
    cos_lat = math.cos(math.radians(avg_lat))
    road_coords_scaled = road_coords.copy()
    road_coords_scaled[:, 0] *= cos_lat
    tile_coords_scaled = tile_coords.copy()
    tile_coords_scaled[:, 0] *= cos_lat
    
    # Build KD-Trees
    tile_tree = cKDTree(tile_coords_scaled)
    road_tree = cKDTree(road_coords_scaled)
    
    road_assignments = defaultdict(float)
    tiles_used = set()
    
    # Pass 1: each road -> nearest tile
    distances, indices = tile_tree.query(road_coords_scaled, k=1)
    
    for road_idx, (dist, tile_idx) in enumerate(zip(distances, indices)):
        if far_thresh_deg is not None and dist > far_thresh_deg:
            continue
        tiles_used.add(tile_idx)
        n = index_to_node[road_idx]
        road_assignments[n] += tile_pops[tile_idx]
    
    # Pass 2: remaining tiles -> nearest road
    tiles_remaining = [i for i in range(len(tile_keys)) if i not in tiles_used]
    tiles_used_pass2 = set()
    
    if tiles_remaining:
        remaining_coords = tile_coords_scaled[tiles_remaining]
        distances, indices = road_tree.query(remaining_coords, k=1)
        
        for i, (dist, road_idx) in enumerate(zip(distances, indices)):
            if far_thresh_deg is not None and dist > far_thresh_deg:
                continue
            tile_idx = tiles_remaining[i]
            tiles_used_pass2.add(tile_idx)
            n = index_to_node[road_idx]
            road_assignments[n] += tile_pops[tile_idx]
    
    # Assign pop_density
    total_pop_assigned = 0.0
    for n, pop in road_assignments.items():
        G.nodes[n]['pop_density'] = pop
        total_pop_assigned += pop
    
    if verbose:
        total_tiles = len(tile_keys)
        tiles_used_total = len(tiles_used) + len(tiles_used_pass2)
        nodes_with_pop = len(road_assignments)
        print(f"Tiles: {tiles_used_total}/{total_tiles} ({100.0*tiles_used_total/total_tiles:.1f}%)")
        print(f"Population: {total_pop_assigned:.0f}/{total_pop_bbox:.0f} ({100.0*total_pop_assigned/total_pop_bbox:.1f}%)")
        print(f"Roads with pop: {nodes_with_pop}/{G.number_of_nodes()}")
    
    return G


def _get_density_slow(G, centers, index_to_node, tile_pop, unique_csv_centers, far_thresh_deg, verbose, total_pop_bbox):
    """Fallback version without scipy."""
    
    csv_centers_unique = [(lon, lat, key) for key, (lon, lat) in unique_csv_centers.items()]

    def _find_nearest_tile(lon, lat):
        best_key, best_lon, best_lat, best_d2 = None, None, None, None
        for tlon, tlat, key in csv_centers_unique:
            d2 = (lon - tlon)**2 + (lat - tlat)**2
            if best_d2 is None or d2 < best_d2:
                best_d2, best_key, best_lon, best_lat = d2, key, tlon, tlat
        return best_key, best_lon, best_lat, best_d2

    tiles_used = set()
    road_assignments = defaultdict(float)

    for idx, (lon, lat) in enumerate(centers):
        n = index_to_node[idx]
        key, tlon, tlat, d2 = _find_nearest_tile(lon, lat)
        if key is None:
            continue
        if far_thresh_deg is not None and math.sqrt(d2) > far_thresh_deg:
            continue
        tiles_used.add(key)
        road_assignments[n] += tile_pop.get(key, 0.0)

    def _find_nearest_road(tlon, tlat):
        best_idx, best_d2 = None, None
        for idx, (rlon, rlat) in enumerate(centers):
            d2 = (tlon - rlon)**2 + (tlat - rlat)**2
            if best_d2 is None or d2 < best_d2:
                best_d2, best_idx = d2, idx
        return best_idx, best_d2

    tiles_used_pass2 = set()
    for key in set(tile_pop.keys()) - tiles_used:
        tlon, tlat = unique_csv_centers[key]
        best_idx, d2 = _find_nearest_road(tlon, tlat)
        if best_idx is None:
            continue
        if far_thresh_deg is not None and math.sqrt(d2) > far_thresh_deg:
            continue
        tiles_used_pass2.add(key)
        n = index_to_node[best_idx]
        road_assignments[n] += tile_pop.get(key, 0.0)

    total_pop_assigned = 0.0
    for n, pop in road_assignments.items():
        G.nodes[n]['pop_density'] = pop
        total_pop_assigned += pop

    if verbose:
        total_tiles = len(tile_pop)
        tiles_used_total = len(tiles_used) + len(tiles_used_pass2)
        nodes_with_pop = len(road_assignments)
        print(f"Tiles: {tiles_used_total}/{total_tiles} ({100.0*tiles_used_total/total_tiles:.1f}%)")
        print(f"Population: {total_pop_assigned:.0f}/{total_pop_bbox:.0f} ({100.0*total_pop_assigned/total_pop_bbox:.1f}%)")
        print(f"Roads with pop: {nodes_with_pop}/{G.number_of_nodes()}")

    return G
