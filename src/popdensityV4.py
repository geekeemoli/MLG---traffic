"""
Center-based population density mapper (v4).

This module assigns population values from a large CSV to each road (line-graph
node). Unlike V3 which only maps each road to its single nearest tile, V4 uses
a two-pass approach:

Pass 1 (road → tile):
  - Same as V3: each road is assigned to its nearest CSV tile center.

Pass 2 (tile → road):
  - For every CSV tile that was NOT used in pass 1, find the nearest road
    within far_thresh_m and assign that tile's population to the road.
  - This ensures that tiles near long roads (where the road center is far
    from multiple tiles) still contribute their population.

The result is that a single road can accumulate population from multiple tiles,
improving coverage of the CSV population data.

Usage:
    from src.popdensityV4 import get_density
    G = get_density(G, csv_path=None)

The function modifies `G` in-place and returns it.
"""

import os
import csv
import math
from collections import defaultdict


def get_density(G, csv_path, tile_half_ddeg=1.0/7200.0, assume_sorted_by_lat=True, far_thresh_m=100.0):
    """
    Annotate graph `G` nodes with 'pop_density' using a two-pass center-point mapping.

    Parameters
    ----------
    G : networkx.Graph
        Line-graph where each node represents a road and has at least a center
        coordinate. If nodes have a 'geometry' attribute (LineString), the centroid
        is used. Otherwise, attempts to use 'x' and 'y' node attributes.
    csv_path : str or None
        Path to population CSV.
    tile_half_ddeg : float
        Half tile size in decimal degrees. Default corresponds to 1 arc-second
        tiles (1/7200 degree half-width).
    assume_sorted_by_lat : bool
        If True, and if the CSV is sorted by latitude ascending, the code will
        perform an early break once lat goes beyond the city's north bound.
    far_thresh_m : float or None
        Maximum distance (meters) for a road-tile assignment. In pass 1, road-to-tile
        assignments farther than this are still made but the tile is also eligible
        for pass 2 reassignment. In pass 2, only tiles within this distance of a
        road can be assigned. If None, no distance limit is applied.

    Returns
    -------
    G : networkx.Graph
        The same graph object, with node attributes updated. Adds key:
          - 'pop_density' (float) : total population assigned to this road
    """

    # --- helpers ---
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

    def _haversine_m(lon1, lat1, lon2, lat2):
        R = 6371000.0
        lon1r, lat1r = math.radians(lon1), math.radians(lat1)
        lon2r, lat2r = math.radians(lon2), math.radians(lat2)
        dlon = lon2r - lon1r
        dlat = lat2r - lat1r
        a = math.sin(dlat/2.0)**2 + math.cos(lat1r)*math.cos(lat2r)*math.sin(dlon/2.0)**2
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0-a)))
        return R * c

    # --- check if already annotated ---
    if any('pop_density' in d for _, d in G.nodes(data=True)):
        print("Graph already has 'pop_density' attribute. Stopping function.")
        return G

    # --- build road centers list ---
    centers = []           # (lon, lat)
    index_to_node = []     # index -> node id
    node_to_index = {}

    for n, data in G.nodes(data=True):
        # initialize attributes
        data['pop_density'] = 0.0

        center = _center_from_node_data(data)
        if center is None:
            continue
        idx = len(centers)
        centers.append(center)
        index_to_node.append(n)
        node_to_index[n] = idx

    if not centers:
        print('No center coordinates found on nodes; nothing to do.')
        return G

    lons = [c[0] for c in centers]
    lats = [c[1] for c in centers]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)

    # Expand bbox to capture edge tiles
    west = min_lon - tile_half_ddeg
    east = max_lon + tile_half_ddeg
    south = min_lat - tile_half_ddeg
    north = max_lat + tile_half_ddeg

    print(f'city centers bbox (deg): lon {min_lon:.6f}..{max_lon:.6f}, lat {min_lat:.6f}..{max_lat:.6f}')
    print(f'processing CSV only inside extended bbox: west={west:.6f}, east={east:.6f}, south={south:.6f}, north={north:.6f}')

    # --- read CSV tiles inside bbox ---

    tile_pop = {}          # key (raw_lon, raw_lat) -> population
    csv_centers = []       # list of (lon_float, lat_float, key)

    try:
        with open(csv_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            lon_field, lat_field, pop_field = reader.fieldnames[:3]

            seen_in_range = False
            rows_processed = 0
            rows_skipped = 0

            for row in reader:
                rows_processed += 1
                try:
                    lon = float(row[lon_field])
                    lat = float(row[lat_field])
                    val = float(row[pop_field])
                except Exception:
                    rows_skipped += 1
                    continue

                if lon < west or lon > east or lat < south or lat > north:
                    if assume_sorted_by_lat:
                        if lat < south:
                            continue
                        if lat > north and seen_in_range:
                            break
                    continue

                seen_in_range = True
                raw_lon = row[lon_field].strip()
                raw_lat = row[lat_field].strip()
                key = (raw_lon, raw_lat)
                tile_pop[key] = tile_pop.get(key, 0.0) + val
                csv_centers.append((lon, lat, key))

            print(f'CSV rows processed: {rows_processed}, rows skipped (parse errors): {rows_skipped}')
            print(f'Unique tiles in bbox: {len(tile_pop)}')

    except FileNotFoundError:
        raise FileNotFoundError(f'Population CSV not found at {csv_path}; please pass csv_path explicitly')

    if not tile_pop:
        print('No CSV tile centers found inside city bbox; nothing to assign.')
        return G

    # deduplicate csv_centers to unique tiles
    unique_csv_centers = {}
    for lon, lat, key in csv_centers:
        if key not in unique_csv_centers:
            unique_csv_centers[key] = (lon, lat)
    csv_centers_unique = [(lon, lat, key) for key, (lon, lat) in unique_csv_centers.items()]

    # --- Pass 1: assign each road to its nearest tile ---
    print('\n--- Pass 1: road -> nearest tile ---')

    def _find_nearest_tile(lon, lat):
        best_key = None
        best_lon = None
        best_lat = None
        best_d2 = None
        for tlon, tlat, key in csv_centers_unique:
            dx = lon - tlon
            dy = lat - tlat
            d2 = dx*dx + dy*dy
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                best_key = key
                best_lon = tlon
                best_lat = tlat
        return best_key, best_lon, best_lat

    tiles_used_pass1 = set()
    road_tile_assignments = defaultdict(list)  # node -> list of (key, lon, lat, dist_m, pop)

    for idx, (lon, lat) in enumerate(centers):
        n = index_to_node[idx]
        key, tlon, tlat = _find_nearest_tile(lon, lat)
        if key is None:
            continue
        dist_m = _haversine_m(lon, lat, tlon, tlat)
        pop = tile_pop.get(key, 0.0)

        # apply far_thresh_m: only accept assignment if within threshold
        if far_thresh_m is not None and dist_m > far_thresh_m:
            # tile not assigned in pass 1, remains available for pass 2
            continue

        tiles_used_pass1.add(key)
        road_tile_assignments[n].append((key, tlon, tlat, dist_m, pop))

    print(f'Pass 1: {len(tiles_used_pass1)} tiles assigned to roads')

    # --- Pass 2: assign remaining tiles to nearest road within threshold ---
    print('\n--- Pass 2: unused tile -> nearest road ---')

    tiles_remaining = set(tile_pop.keys()) - tiles_used_pass1
    print(f'Tiles remaining after pass 1: {len(tiles_remaining)}')

    def _find_nearest_road(tlon, tlat):
        best_idx = None
        best_d2 = None
        for idx, (rlon, rlat) in enumerate(centers):
            dx = tlon - rlon
            dy = tlat - rlat
            d2 = dx*dx + dy*dy
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                best_idx = idx
        return best_idx

    tiles_assigned_pass2 = 0
    tiles_too_far_pass2 = 0

    for key in tiles_remaining:
        tlon, tlat = unique_csv_centers[key]
        best_idx = _find_nearest_road(tlon, tlat)
        if best_idx is None:
            continue

        rlon, rlat = centers[best_idx]
        dist_m = _haversine_m(tlon, tlat, rlon, rlat)

        # apply far_thresh_m
        if far_thresh_m is not None and dist_m > far_thresh_m:
            tiles_too_far_pass2 += 1
            continue

        n = index_to_node[best_idx]
        pop = tile_pop.get(key, 0.0)
        road_tile_assignments[n].append((key, tlon, tlat, dist_m, pop))
        tiles_assigned_pass2 += 1

    print(f'Pass 2: {tiles_assigned_pass2} additional tiles assigned to roads')
    print(f'Pass 2: {tiles_too_far_pass2} tiles too far (>{far_thresh_m}m), not assigned')

    # --- aggregate and write node attributes ---
    total_tiles_used = len(tiles_used_pass1) + tiles_assigned_pass2
    print(f'\nTotal tiles used: {total_tiles_used} out of {len(tile_pop)} ({100.0*total_tiles_used/len(tile_pop):.2f}%)')

    nodes_with_tiles = 0
    total_pop_assigned = 0.0

    # track tile -> roads mapping for analysis
    tile_to_roads = defaultdict(list)

    for n, assignments in road_tile_assignments.items():
        if not assignments:
            continue

        nodes_with_tiles += 1
        total_pop = sum(pop for key, tlon, tlat, dist_m, pop in assignments)
        total_pop_assigned += total_pop

        G.nodes[n]['pop_density'] = total_pop

        # record which roads each tile is assigned to
        for key, tlon, tlat, dist_m, pop in assignments:
            tile_to_roads[key].append((n, dist_m))

    print(f'Nodes with at least one tile: {nodes_with_tiles} out of {G.number_of_nodes()}')
    print(f'Total population assigned to roads: {total_pop_assigned:.3f}')

    # compute total population in bbox for comparison
    total_pop_bbox = sum(tile_pop.values())
    print(f'Total population in CSV bbox: {total_pop_bbox:.3f}')
    print(f'Coverage: {100.0*total_pop_assigned/total_pop_bbox:.2f}%')

    # Store analysis data as graph attribute for later use
    G.graph['_v4_road_tile_assignments'] = dict(road_tile_assignments)
    G.graph['_v4_tile_to_roads'] = dict(tile_to_roads)
    G.graph['_v4_unique_csv_centers'] = unique_csv_centers
    G.graph['_v4_tile_pop'] = tile_pop

    return G


def analyze_density(G, top_n=100, verbose=True):
    """
    Analyze and print basic statistics about the V4 population assignment.

    Returns a dict with summary statistics for programmatic checks.
    """
    import statistics

    total_nodes = G.number_of_nodes()
    nodes_with_pop = 0
    values = []

    for n, d in G.nodes(data=True):
        val = d.get('pop_density', 0.0)
        if val > 0:
            nodes_with_pop += 1
            values.append(float(val))

    nodes_zero = total_nodes - nodes_with_pop

    summary = {
        'total_nodes': total_nodes,
        'nodes_with_pop': nodes_with_pop,
        'nodes_zero_pop': nodes_zero,
    }

    if values:
        summary.update({
            'pop_min': min(values),
            'pop_max': max(values),
            'pop_mean': statistics.mean(values),
            'pop_median': statistics.median(values),
            'pop_stdev': statistics.pstdev(values) if len(values) > 1 else 0.0,
            'pop_total': sum(values),
        })

    if verbose:
        print('--- pop_density analysis (V4) ---')
        print(f"total nodes: {total_nodes}")
        print(f"nodes with pop > 0: {nodes_with_pop}")
        print(f"nodes with pop = 0: {nodes_zero}")

        if values:
            print(f"\npop_density stats (non-zero only):")
            print(f"  min={summary['pop_min']:.3f}, max={summary['pop_max']:.3f}")
            print(f"  mean={summary['pop_mean']:.3f}, median={summary['pop_median']:.3f}, stdev={summary['pop_stdev']:.3f}")
            print(f"  total={summary['pop_total']:.3f}")

        # top nodes by pop_density
        top_nodes = sorted(
            ((n, d.get('pop_density', 0.0)) for n, d in G.nodes(data=True)),
            key=lambda x: x[1], reverse=True
        )[:top_n]
        print(f'\nTop {top_n} nodes by pop_density:')
        for n, pop in top_nodes:
            print(f"  node {n} -> pop={pop:.3f}")

    return summary


def export_top_roads_tiles_geojson(G, output_dir, top_n=100, city_name='city'):
    """
    Export GeoJSON files for:
      1. Top roads by number of tiles assigned
      2. Top tiles by number of roads assigned

    Parameters
    ----------
    G : networkx.Graph
        Graph that has been processed by get_density (contains _v4_* graph attributes).
    output_dir : str
        Directory to write GeoJSON files.
    top_n : int
        Number of top entries to export.
    city_name : str
        City name for file naming.
    """
    import json
    import os

    # Helper to get node center
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
        return (None, None)

    os.makedirs(output_dir, exist_ok=True)
    city_tag = city_name.replace(",", "_").replace(" ", "_")

    road_tile_assignments = G.graph.get('_v4_road_tile_assignments', {})
    tile_to_roads = G.graph.get('_v4_tile_to_roads', {})
    unique_csv_centers = G.graph.get('_v4_unique_csv_centers', {})
    tile_pop = G.graph.get('_v4_tile_pop', {})

    if not road_tile_assignments:
        print('No assignment data found. Run get_density first.')
        return

    # --- 1. Top roads by tile count ---
    roads_by_tile_count = sorted(
        ((n, len(assignments), assignments) for n, assignments in road_tile_assignments.items()),
        key=lambda x: x[1], reverse=True
    )[:top_n]

    top_roads_geo = {'type': 'FeatureCollection', 'features': []}

    for n, tile_count, assignments in roads_by_tile_count:
        data = G.nodes[n]
        center = _center_from_node_data(data)
        if center[0] is None or center[1] is None:
            continue

        total_pop = sum(pop for key, tlon, tlat, dist_m, pop in assignments)
        tile_keys = [key for key, tlon, tlat, dist_m, pop in assignments]
        avg_dist = sum(dist_m for key, tlon, tlat, dist_m, pop in assignments) / tile_count if tile_count > 0 else 0

        props = {
            'node_id': str(n),
            'tile_count': tile_count,
            'pop_density': total_pop,
            'avg_tile_distance_m': round(avg_dist, 2),
            'tile_keys': [f"{k[0]},{k[1]}" for k in tile_keys[:10]],  # limit to first 10
        }

        # Add road geometry if available
        geom = data.get('geometry')
        if geom is not None:
            try:
                coords = list(geom.coords)
                feat = {
                    'type': 'Feature',
                    'geometry': {'type': 'LineString', 'coordinates': [[c[0], c[1]] for c in coords]},
                    'properties': props
                }
            except Exception:
                feat = {
                    'type': 'Feature',
                    'geometry': {'type': 'Point', 'coordinates': [center[0], center[1]]},
                    'properties': props
                }
        else:
            feat = {
                'type': 'Feature',
                'geometry': {'type': 'Point', 'coordinates': [center[0], center[1]]},
                'properties': props
            }

        top_roads_geo['features'].append(feat)

    top_roads_path = os.path.join(output_dir, f'top_roads_by_tiles_{city_tag}.geojson')
    with open(top_roads_path, 'w', encoding='utf-8') as f:
        json.dump(top_roads_geo, f, ensure_ascii=False, indent=2)
    print(f'Wrote top {len(roads_by_tile_count)} roads by tile count to {top_roads_path}')

    # --- 2. Top tiles by road count ---
    tiles_by_road_count = sorted(
        ((key, len(roads), roads) for key, roads in tile_to_roads.items()),
        key=lambda x: x[1], reverse=True
    )[:top_n]

    top_tiles_geo = {'type': 'FeatureCollection', 'features': []}

    for key, road_count, roads in tiles_by_road_count:
        if key not in unique_csv_centers:
            continue
        tlon, tlat = unique_csv_centers[key]
        pop = tile_pop.get(key, 0.0)
        avg_dist = sum(dist_m for n, dist_m in roads) / road_count if road_count > 0 else 0

        props = {
            'tile_key': f"{key[0]},{key[1]}",
            'tile_lon': tlon,
            'tile_lat': tlat,
            'tile_pop': pop,
            'road_count': road_count,
            'avg_road_distance_m': round(avg_dist, 2),
            'sample_road_ids': [str(n) for n, dist_m in roads[:10]],  # limit to first 10
        }

        feat = {
            'type': 'Feature',
            'geometry': {'type': 'Point', 'coordinates': [tlon, tlat]},
            'properties': props
        }
        top_tiles_geo['features'].append(feat)

    top_tiles_path = os.path.join(output_dir, f'top_tiles_by_roads_{city_tag}.geojson')
    with open(top_tiles_path, 'w', encoding='utf-8') as f:
        json.dump(top_tiles_geo, f, ensure_ascii=False, indent=2)
    print(f'Wrote top {len(tiles_by_road_count)} tiles by road count to {top_tiles_path}')


if __name__ == '__main__':
    print('This module provides get_density(G, csv_path=None, tile_half_ddeg=..., far_thresh_m=...)')
    print('V4: two-pass assignment (road->tile, then unused tile->nearest road)')
