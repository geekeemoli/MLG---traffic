"""
Center-based population density mapper (v3).

This module assigns a population value from a large CSV to each road (line-graph
node) by using the road's center point instead of spatial intersection tests.

Main features:
- compute center point (lon, lat) for each road node (from 'geometry' or 'x'/'y')
- compute bounding box of the city (centers) and restrict CSV processing to tiles
  that intersect this bbox (speeds up processing on country-wide CSV)
- map roads to a tile via rounding to the tile grid, accumulate tile population,
  count roads per tile and then normalize (each road gets tile_population / road_count)
- configurable tile half-width in degrees and an option to assume CSV is sorted by
  latitude (allows early break)

Usage:
    from src.popdensityV3 import get_density
    G = get_density(G, csv_path=None)

The function modifies `G` in-place and returns it.
"""

import os
import csv
import math
from collections import defaultdict
from shapely.geometry import Point

def get_density(G, csv_path=None, tile_half_ddeg=1.0/7200.0, assume_sorted_by_lat=True, far_thresh_m=75.0):
    """
    Annotate graph `G` nodes with 'pop_density' using center-point mapping.

    Parameters
    ----------
    G : networkx.Graph
        Line-graph where each node represents a road and has at least a center
        coordinate. If nodes have a 'geometry' attribute (LineString), the centroid
        is used. Otherwise, attempts to use 'x' and 'y' node attributes.
    csv_path : str or None
        Path to population CSV. If None, uses ../data/population_data/aut_general_2020.csv
        relative to this file.
    tile_half_ddeg : float
        Half tile size in decimal degrees. Default corresponds to 1 arc-second
        tiles (1/7200 degree half-width).
    assume_sorted_by_lat : bool
        If True, and if the CSV is sorted by latitude ascending, the code will
        perform an early break once lat goes beyond the city's north bound.

    Parameters
    ----------
    far_thresh_m : float or None
        If a float (meters) is provided, any node whose assigned tile is farther
        than this distance will have its 'pop_density' set to 0.0. If None, no
        distance-based zeroing is applied. Default is 1000.0 (1 km) to preserve
        previous behaviour.

    Returns
    -------
    G : networkx.Graph
        The same graph object, with node attributes updated. Adds keys:
          - 'pop_density' (float) : normalized per-road density assigned
          - 'tile_center' (tuple) : (lon, lat) center of assigned tile
          - 'raw_tile_pop' (float) : total population value of assigned tile
    """

    # small helper
    def _center_from_node_data(data):
        # prefer explicit centroid geometry
        geom = data.get('geometry')
        if geom is not None:
            try:
                c = geom.centroid
                return float(c.x), float(c.y)
            except Exception:
                pass
        # fallback to attributes
        x = data.get('x') or data.get('lon') or data.get('long')
        y = data.get('y') or data.get('lat')
        if x is not None and y is not None:
            return float(x), float(y)
        return None

    # if graph already annotated, skip
    if any('pop_density' in d for _, d in G.nodes(data=True)):
        print("Graph already has 'pop_density' attribute. Stopping function.")
        return G

    # prepare centers and mapping
    tile_size = 2.0 * float(tile_half_ddeg)

    centers = []            # list of (lon, lat)
    index_to_node = []      # index -> node id
    node_to_index = {}

    for n, data in G.nodes(data=True):
        data['pop_density'] = 0.0
        center = _center_from_node_data(data)
        if center is None:
            # leave node with pop_density 0.0 but continue
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

    # Expand bbox by half tile to capture any edge tiles
    west = min_lon - tile_half_ddeg
    east = max_lon + tile_half_ddeg
    south = min_lat - tile_half_ddeg
    north = max_lat + tile_half_ddeg

    print(f'city centers bbox (deg): lon {min_lon:.6f}..{max_lon:.6f}, lat {min_lat:.6f}..{max_lat:.6f}')
    print(f'processing CSV only inside extended bbox: west={west:.6f}, east={east:.6f}, south={south:.6f}, north={north:.6f}')

    # We'll first scan the CSV and collect tile centers and populations that fall
    # inside the city bbox. From those tile centers we derive an origin so that
    # snapping road centers to the CSV grid aligns correctly with the CSV tile
    # centers. This avoids origin mismatch that caused many roads to be unassigned.

    # accumulate tile populations from CSV (collect all tile centers in bbox first)
    if csv_path is None:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        csv_path = os.path.join(repo_root, 'data', 'population_data', 'aut_general_2020.csv')

    tile_pop = defaultdict(float)  # key -> sum of values (key will be CSV string pair)
    seen_tile_lons = []
    seen_tile_lats = []
    csv_centers = []  # list of (lon_float, lat_float, key)

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

                # quickly skip rows outside our bbox
                if lon < west or lon > east or lat < south or lat > north:
                    # if CSV sorted by lat ascending we can do an early skip/break on lat
                    if assume_sorted_by_lat:
                        # if lat < south -> continue (file has smaller lat values first)
                        if lat < south:
                            continue
                        # if lat > north and we've already seen relevant rows, we can break
                        if lat > north and seen_in_range:
                            break
                    continue

                # we are inside city-extended bbox
                seen_in_range = True

                # preserve raw CSV strings as key to avoid precision loss
                raw_lon = row[lon_field].strip()
                raw_lat = row[lat_field].strip()
                # parse floats for computations
                lon = float(raw_lon)
                lat = float(raw_lat)
                key = (raw_lon, raw_lat)
                tile_pop[key] += val
                seen_tile_lons.append(lon)
                seen_tile_lats.append(lat)
                csv_centers.append((lon, lat, key))

            print(f'CSV rows processed: {rows_processed}, rows skipped (parse errors): {rows_skipped}')

    except FileNotFoundError:
        raise FileNotFoundError(f'Population CSV not found at {csv_path}; please pass csv_path explicitly')

    # if we didn't find any CSV tiles in the bbox, warn and return unchanged graph
    if not tile_pop:
        print('No CSV tile centers found inside city bbox; nothing to assign.')
        return G

    # Build a nearest-lookup structure for CSV tile centers (list of floats + key)
    # csv_centers is already filled during CSV reading as (lon, lat, key)

    def _find_nearest_csv_center_key(lon, lat):
        # Linear scan nearest neighbor; returns the CSV key (raw string pair)
        best = None
        best_d2 = None
        for cl, ct, key in csv_centers:
            dx = lon - cl
            dy = lat - ct
            d2 = dx*dx + dy*dy
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                best = (cl, ct, key)
        if best is None:
            return None, None, None
        return best[2], best[0], best[1]

    # Build mapping from tile keys (CSV-aligned centers) to node lists and record
    # per-node assignment metadata (method and distance). Here we assign every
    # node to the nearest CSV tile unconditionally.
    tile_to_nodes = defaultdict(list)
    node_assignment = {}  # node -> {'tile': key, 'method': 'exact'|'nearest', 'dist_deg': ..., 'dist_m': ...}

    def _haversine_m(lon1, lat1, lon2, lat2):
        # return distance in meters between two lon/lat points (degrees)
        R = 6371000.0
        lon1r = math.radians(lon1)
        lat1r = math.radians(lat1)
        lon2r = math.radians(lon2)
        lat2r = math.radians(lat2)
        dlon = lon2r - lon1r
        dlat = lat2r - lat1r
        a = math.sin(dlat/2.0)**2 + math.cos(lat1r) * math.cos(lat2r) * math.sin(dlon/2.0)**2
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0-a)))
        return R * c

    for idx, (lon, lat) in enumerate(centers):
        n = index_to_node[idx]
        # find nearest CSV tile center (unconditional)
        key, cl, ct = _find_nearest_csv_center_key(lon, lat)
        if key is None:
            node_assignment[n] = {'tile': None, 'method': 'none', 'dist_deg': None, 'dist_m': None}
            continue

        tile_to_nodes[key].append(n)
        # compute distance
        dx = lon - float(cl)
        dy = lat - float(ct)
        dist_deg = math.hypot(dx, dy)
        dist_m = _haversine_m(lon, lat, float(cl), float(ct))
        # if the snapped grid matched exactly to CSV center (very unlikely), mark exact
        method = 'exact' if dist_deg == 0.0 else 'nearest'
        node_assignment[n] = {'tile': key, 'method': method, 'dist_deg': dist_deg, 'dist_m': dist_m}

    # assign normalized pop_density to nodes and record assignment metadata
    assigned_nonzero = 0
    for key, nodes in tile_to_nodes.items():
        pop_sum = tile_pop.get(key, 0.0)
        per_road = pop_sum / float(len(nodes)) if pop_sum != 0.0 else 0.0
        for n in nodes:
            # assignment metadata
            meta = node_assignment.get(n, {})
            method = meta.get('method', 'unknown')
            dist_deg = meta.get('dist_deg')
            dist_m = meta.get('dist_m')

            G.nodes[n]['pop_density'] = per_road
            G.nodes[n]['tile_center'] = key
            G.nodes[n]['raw_tile_pop'] = pop_sum
            G.nodes[n]['assigned_by'] = method
            G.nodes[n]['tile_distance_deg'] = dist_deg
            G.nodes[n]['tile_distance_m'] = dist_m

            if per_road != 0.0:
                assigned_nonzero += 1

    total_nodes = G.number_of_nodes()
    print(f'assigned pop_density>0 to {assigned_nonzero} nodes out of {total_nodes} (total nodes)')

    # Optionally zero-out nodes assigned to tiles that are too far away
    zeroed_far = 0
    if far_thresh_m is not None:
        far_nodes = [(n, d.get('tile_distance_m')) for n, d in G.nodes(data=True) if d.get('tile_distance_m') and d.get('tile_distance_m') > float(far_thresh_m)]
        if far_nodes:
            far_nodes.sort(key=lambda x: x[1], reverse=True)
            print(f'Warning: {len(far_nodes)} nodes assigned to tiles farther than {far_thresh_m} m. Top 5:')
            for n, dist in far_nodes[:5]:
                print(f' node {n} -> {dist:.1f} m')

        # apply zeroing
        for n, d in G.nodes(data=True):
            td = d.get('tile_distance_m')
            if td is not None and td > float(far_thresh_m):
                d['pop_density'] = 0.0
                # mark assignment as too_far; preserve existing method tag if any
                prev = d.get('assigned_by')
                d['assigned_by'] = (prev + '|too_far') if prev else 'too_far'
                zeroed_far += 1

        if zeroed_far:
            print(f'Applied far_thresh_m={far_thresh_m}: set pop_density=0.0 for {zeroed_far} nodes.')

    else:
        # if thresholding disabled, still report distant nodes for diagnostics
        far_nodes = [(n, d.get('tile_distance_m')) for n, d in G.nodes(data=True) if d.get('tile_distance_m') and d.get('tile_distance_m') > 1000.0]
        if far_nodes:
            far_nodes.sort(key=lambda x: x[1], reverse=True)
            print(f'Info: {len(far_nodes)} nodes assigned to tiles farther than 1000.0 m (no threshold applied). Top 5:')
            for n, dist in far_nodes[:5]:
                print(f' node {n} -> {dist:.1f} m')

    # recompute assigned_nonzero after possible zeroing
    assigned_nonzero = sum(1 for _, d in G.nodes(data=True) if float(d.get('pop_density', 0.0)) != 0.0)

    total_nodes = G.number_of_nodes()
    print(f'assigned pop_density>0 to {assigned_nonzero} nodes out of {total_nodes} (total nodes)')
    return G


if __name__ == '__main__':
    print('This module provides get_density(G, csv_path=None, tile_half_ddeg=...)')


def analyze_density(G, top_n=10, verbose=True):
    """
    Analyze and print basic statistics about the population assignment.

    Returns a dict with summary statistics for programmatic checks.
    """
    import statistics

    total_nodes = G.number_of_nodes()
    assigned_nodes = 0
    values = []
    tile_groups = {}
    exact_count = 0
    nearest_count = 0
    none_count = 0
    far_list = []

    for n, d in G.nodes(data=True):
        if 'tile_center' in d:
            assigned_nodes += 1
            val = float(d.get('pop_density', 0.0))
            values.append(val)
            key = tuple(d.get('tile_center'))
            if key not in tile_groups:
                tile_groups[key] = {'nodes': [], 'tile_pop': float(d.get('raw_tile_pop', 0.0))}
            tile_groups[key]['nodes'].append(n)
            method = d.get('assigned_by')
            if method == 'exact':
                exact_count += 1
            elif method == 'nearest':
                nearest_count += 1
            else:
                none_count += 1
            td = d.get('tile_distance_m')
            if td:
                far_list.append((n, td))

    unassigned = total_nodes - assigned_nodes
    unassigned_nodes_list = [n for n,d in G.nodes(data=True) if d.get('tile_center') is None]

    summary = {
        'total_nodes': total_nodes,
        'assigned_nodes': assigned_nodes,
        'unassigned_nodes': unassigned,
        'tiles_with_roads': len(tile_groups),
        'unassigned_nodes_list': unassigned_nodes_list,
    }

    if values:
        summary.update({
            'min': min(values),
            'max': max(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'stdev': statistics.pstdev(values) if len(values) > 1 else 0.0,
        })

    # aggregate per-tile checks
    tiles_info = []
    total_tile_pop = 0.0
    for key, info in tile_groups.items():
        tp = float(info.get('tile_pop', 0.0))
        nr = len(info['nodes'])
        total_tile_pop += tp
        tiles_info.append((key, tp, nr))

    tiles_info.sort(key=lambda x: x[1], reverse=True)

    # sum of assigned node pop_density
    sum_assigned = sum(float(d.get('pop_density', 0.0)) for _, d in G.nodes(data=True) if 'tile_center' in d)

    summary.update({'total_tile_pop': total_tile_pop, 'sum_assigned_node_pop': sum_assigned})

    if verbose:
        print('--- pop_density analysis ---')
        print(f"total nodes: {total_nodes}")
        print(f"assigned nodes: {assigned_nodes}")
        print(f"unassigned nodes: {unassigned}")
        print(f"tiles with roads: {len(tile_groups)}")
        print(f"assignment methods: exact={exact_count}, nearest={nearest_count}, none={none_count}")
        if values:
            print(f"pop_density: min={summary['min']:.3f}, max={summary['max']:.3f}, mean={summary['mean']:.3f}, median={summary['median']:.3f}, stdev={summary['stdev']:.3f}")
        print(f"total tile population (used tiles): {total_tile_pop:.3f}")
        print(f"sum of assigned node pop_density: {sum_assigned:.3f}")
        print(f"relative difference: {(sum_assigned - total_tile_pop):.6f} (should be ~0)")

        print('\nTop tiles by population:')
        for key, tp, nr in tiles_info[:top_n]:
            print(f" tile {key} -> pop={tp:.1f}, roads={nr}, per-road~{tp/float(nr):.2f}")


        # nodes with highest pop_density
        top_nodes = sorted(((n, d.get('pop_density', 0.0)) for n, d in G.nodes(data=True)), key=lambda x: x[1], reverse=True)[:top_n]
        print('\nTop nodes by pop_density:')
        for n, v in top_nodes:
            print(f" node {n} -> {v}")

        # farthest assigned nodes with details: include node center, assigned tile center and assigned pop_density
        if far_list:
            far_list.sort(key=lambda x: x[1], reverse=True)
            print('\nTop nodes by tile-distance (meters) with details:')

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

            for n, td in far_list[:top_n]:
                d = G.nodes[n]
                node_center = _center_from_node_data(d)
                tile_center = d.get('tile_center')
                pd = d.get('pop_density')
                print(f" node {n} -> dist={td:.1f} m, center={node_center}, tile={tile_center}, pop_density={pd}")

        # print unassigned node ids (first 20) so user can inspect
        if unassigned_nodes_list:
            print('\nUnassigned node ids (lack tile_center) - first 20:')
            print(unassigned_nodes_list[:20])

    return summary
