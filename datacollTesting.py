import os
import csv
import osmnx as ox
import networkx as nx
from src.popdensityV3 import *
from shapely.geometry import LineString

cities = ["Graz, Austria"]
graphs = {}

for city in cities:
    print(f"Processing {city}")
    G = ox.graph_from_place(city, network_type="drive")

    # ensure edges have geometry
    for u, v, k, data in G.edges(keys=True, data=True):
        if 'geometry' not in data:
            x1 = G.nodes[u].get('x'); y1 = G.nodes[u].get('y')
            x2 = G.nodes[v].get('x'); y2 = G.nodes[v].get('y')
            if None not in (x1, y1, x2, y2):
                data['geometry'] = LineString([(x1, y1), (x2, y2)])

    road_G = nx.line_graph(G)

    # copy edge attributes from G to line-graph node attributes
    for u, v, k, data in G.edges(keys=True, data=True):
        lg_node = (u, v, k)
        if lg_node in road_G:
            road_G.nodes[lg_node].update(data)

    graphs[city] = road_G
    print(f"{len(road_G.nodes)} roads, {len(road_G.edges)} adjacencies")

for city in cities:
    print(f'adding density to {city}')
    graph_popd = get_density(graphs[city])
    # quick samples
    # print('sample with pop_density>0:', [n for n,d in graph_popd.nodes(data=True) if d.get('pop_density',0)>0][:8])
    summary = analyze_density(graph_popd, top_n=20, verbose=True)
    # export reports: far-assigned nodes and unassigned nodes
    reports_dir = os.path.join(os.path.dirname(__file__), 'analysis_reports')
    os.makedirs(reports_dir, exist_ok=True)

    # helper to get node center
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

    # thresholds for far-node reporting (meters).
    far_thresholds = [40.0, 50.0, 60.0, 70.0, 80.0]
    # vseh vozlisc: 11256
    # pri 40m: 532 => 0.0472636815920398
    # pri 50m: 315 => 0.0279850746268657
    # pri 60m: 209 => 0.0185678749111585
    # pri 70m: 164 => 0.0145700071073205
    # pri 75m: 147 => 0.0130597014925373
    # pri 80m: 135 => 0.0119936034115139

    # write unassigned nodes (no tile_center) once
    unassigned = [ (n,d) for n,d in graph_popd.nodes(data=True) if d.get('tile_center') is None ]
    un_path = os.path.join(reports_dir, f'unassigned_nodes_{city.replace(",","_").replace(" ","_")}.csv')
    with open(un_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['node_id', 'center_lon', 'center_lat'])
        for n, d in unassigned:
            center = _center_from_node_data(d)
            writer.writerow([n, center[0], center[1]])
    print(f'Wrote {len(unassigned)} unassigned nodes to {un_path}')

    for far_thresh_m in far_thresholds:
        # write far nodes (tile_distance_m > threshold)
        thr_tag = f"thr{int(far_thresh_m)}"
        far_path = os.path.join(reports_dir, f'far_assigned_nodes_{city.replace(",","_").replace(" ","_")}_{thr_tag}.csv')
        with open(far_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['node_id', 'center_lon', 'center_lat', 'tile_center_lon', 'tile_center_lat', 'tile_distance_m', 'assigned_by', 'pop_density', 'raw_tile_pop'])
            count_far = 0
            for n, d in graph_popd.nodes(data=True):
                td = d.get('tile_distance_m')
                if td and td > far_thresh_m:
                    center = _center_from_node_data(d)
                    tile = d.get('tile_center') or ('', '')
                    # tile keys in popdensityV3 are raw strings (lon, lat)
                    tile_lon = tile[0] if tile else ''
                    tile_lat = tile[1] if tile else ''
                    writer.writerow([n, center[0], center[1], tile_lon, tile_lat, td, d.get('assigned_by'), d.get('pop_density'), d.get('raw_tile_pop')])
                    count_far += 1
        print(f'Wrote {count_far} far-assigned nodes (>{far_thresh_m} m) to {far_path}')
        # Group far nodes by their assigned tile and write a tile-level summary for this threshold
        from collections import defaultdict
        by_tile = defaultdict(list)
        for n, d in graph_popd.nodes(data=True):
            td = d.get('tile_distance_m')
            if td and td > far_thresh_m:
                tile = d.get('tile_center') or ('', '')
                center = _center_from_node_data(d)
                by_tile[tile].append({'node': n, 'dist_m': td, 'center': center, 'pop_density': d.get('pop_density'), 'raw_tile_pop': d.get('raw_tile_pop')})

        tile_summary_path = os.path.join(reports_dir, f'far_tile_summary_{city.replace(",","_").replace(" ","_")}_{thr_tag}.csv')
        with open(tile_summary_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['tile_center_lon', 'tile_center_lat', 'tile_pop', 'nodes_count', 'avg_dist_m', 'max_dist_m', 'sample_node_ids'])
            for tile, nodes in sorted(by_tile.items(), key=lambda x: sum(n['dist_m'] for n in x[1]), reverse=True):
                tile_lon = tile[0]
                tile_lat = tile[1]
                node_count = len(nodes)
                avg_dist = sum(n['dist_m'] for n in nodes) / node_count
                max_dist = max(n['dist_m'] for n in nodes)
                tile_pop = nodes[0].get('raw_tile_pop') if nodes else ''
                sample_ids = [n['node'] for n in nodes[:5]]
                writer.writerow([tile_lon, tile_lat, tile_pop, node_count, f"{avg_dist:.3f}", f"{max_dist:.3f}", ";".join(map(str, sample_ids))])
        print(f'Wrote far-tile summary to {tile_summary_path}')

        # Write per-tile detail file listing nodes for each far tile
        tile_detail_path = os.path.join(reports_dir, f'far_nodes_by_tile_{city.replace(",","_").replace(" ","_")}_{thr_tag}.csv')
        with open(tile_detail_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['tile_center_lon', 'tile_center_lat', 'node_id', 'center_lon', 'center_lat', 'tile_distance_m', 'pop_density', 'raw_tile_pop'])
            for tile, nodes in by_tile.items():
                tile_lon = tile[0]
                tile_lat = tile[1]
                for info in nodes:
                    writer.writerow([tile_lon, tile_lat, info['node'], info['center'][0], info['center'][1], f"{info['dist_m']:.3f}", info.get('pop_density'), info.get('raw_tile_pop')])
        print(f'Wrote far-tile detail file to {tile_detail_path}')

        # Export GeoJSON for far nodes and their assigned tile centers for quick mapping
        import json
        far_nodes_geo = {'type': 'FeatureCollection', 'features': []}
        far_tiles_geo = {'type': 'FeatureCollection', 'features': []}

        # Collect unique tiles we've seen in by_tile
        tiles_seen = {}
        for tile, nodes in by_tile.items():
            # tile is a tuple of raw string coords (lon_str, lat_str)
            try:
                tile_lon = float(tile[0])
                tile_lat = float(tile[1])
            except Exception:
                continue
            nodes_count = len(nodes)
            tile_pop = nodes[0].get('raw_tile_pop') if nodes else None
            tiles_seen[tile] = {'lon': tile_lon, 'lat': tile_lat, 'tile_pop': tile_pop, 'nodes_count': nodes_count, 'sample_nodes': [n['node'] for n in nodes[:5]]}

        # far nodes geojson features
        for tile, nodes in by_tile.items():
            for info in nodes:
                node = info['node']
                center = info['center']
                if center[0] is None or center[1] is None:
                    continue
                props = {
                    'node_id': node,
                    'tile_center': tile,
                    'tile_distance_m': float(info['dist_m']) if info['dist_m'] is not None else None,
                    'pop_density': info.get('pop_density'),
                    'raw_tile_pop': info.get('raw_tile_pop'),
                    'assigned_by': graph_popd.nodes[node].get('assigned_by')
                }
                feat = {'type': 'Feature', 'geometry': {'type': 'Point', 'coordinates': [float(center[0]), float(center[1])]}, 'properties': props}
                far_nodes_geo['features'].append(feat)

        # far tiles geojson features
        for tile, meta in tiles_seen.items():
            props = {
                'tile_center': tile,
                'tile_pop': meta.get('tile_pop'),
                'nodes_count': meta.get('nodes_count'),
                'sample_nodes': meta.get('sample_nodes')
            }
            feat = {'type': 'Feature', 'geometry': {'type': 'Point', 'coordinates': [meta['lon'], meta['lat']]}, 'properties': props}
            far_tiles_geo['features'].append(feat)

        far_nodes_geo_path = os.path.join(reports_dir, f'far_nodes_{city.replace(",","_").replace(" ","_")}_{thr_tag}.geojson')
        far_tiles_geo_path = os.path.join(reports_dir, f'far_tiles_{city.replace(",","_").replace(" ","_")}_{thr_tag}.geojson')
        with open(far_nodes_geo_path, 'w', encoding='utf-8') as f:
            json.dump(far_nodes_geo, f, ensure_ascii=False, indent=2)
        with open(far_tiles_geo_path, 'w', encoding='utf-8') as f:
            json.dump(far_tiles_geo, f, ensure_ascii=False, indent=2)

        print(f'Wrote far nodes GeoJSON to {far_nodes_geo_path}')
        print(f'Wrote far tiles GeoJSON to {far_tiles_geo_path}')
    # Combined summary across thresholds for quick comparison
    combined_path = os.path.join(reports_dir, f'far_thresholds_summary_{city.replace(",","_").replace(" ","_")}.csv')
    with open(combined_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['threshold_m', 'nodes_count', 'distinct_tiles', 'avg_dist_m', 'max_dist_m', 'pct_of_total_nodes'])
        total_nodes = graph_popd.number_of_nodes()
        for far_thresh_m in far_thresholds:
            nodes = [d for _,d in graph_popd.nodes(data=True) if (d.get('tile_distance_m') and d.get('tile_distance_m')>far_thresh_m)]
            count_nodes = len(nodes)
            distinct_tiles = len({d.get('tile_center') for d in nodes if d.get('tile_center') is not None})
            dists = [d.get('tile_distance_m') for d in nodes if d.get('tile_distance_m') is not None]
            avg_dist = (sum(dists)/len(dists)) if dists else 0.0
            max_dist = max(dists) if dists else 0.0
            pct = 100.0 * count_nodes / float(total_nodes) if total_nodes>0 else 0.0
            writer.writerow([far_thresh_m, count_nodes, distinct_tiles, f"{avg_dist:.3f}", f"{max_dist:.3f}", f"{pct:.3f}"])
    print(f'Wrote combined thresholds summary to {combined_path}')


