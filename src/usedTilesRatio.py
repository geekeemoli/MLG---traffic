import os
import csv
import osmnx as ox
import networkx as nx
from popdensityV3 import *
from shapely.geometry import LineString

#output
# Processing Graz, Austria
# 11256 roads, 30845 adjacencies
#CSV tiles in bbox: 73758
# Tiles used by roads: 5693
# Fraction used: 0.07718484774532933

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
csv_path = os.path.join(repo_root, 'data', 'population_data', 'aut_general_2020.csv')

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

    # annotate and analyze the line-graph
    road_G = get_density(road_G, csv_path=csv_path, far_thresh_m=75.0)

    graphs[city] = road_G
    print(f"{len(road_G.nodes)} roads, {len(road_G.edges)} adjacencies")

# compute bbox from G (same as get_density)
lons = [d.get('x') or d.get('lon') or (d.get('geometry').centroid.x if d.get('geometry') is not None else None) for _,d in G.nodes(data=True)]
lats = [d.get('y') or d.get('lat') or (d.get('geometry').centroid.y if d.get('geometry') is not None else None) for _,d in G.nodes(data=True)]
lons = [float(v) for v in lons if v is not None]
lats = [float(v) for v in lats if v is not None]
tile_half = 1.0/7200.0
west, east = min(lons) - tile_half, max(lons) + tile_half
south, north = min(lats) - tile_half, max(lats) + tile_half

csv_keys = set()
with open(csv_path, newline='', encoding='utf-8') as f:
    r = csv.DictReader(f)
    lon_field, lat_field, pop_field = r.fieldnames[:3]
    for row in r:
        try:
            lon = float(row[lon_field]); lat = float(row[lat_field])
        except Exception:
            continue
        if lon < west or lon > east or lat < south or lat > north:
            continue
        csv_keys.add((row[lon_field].strip(), row[lat_field].strip()))

# compute used keys from the road graph (line-graph), not the original G
used_keys = set(d.get('tile_center') for _, d in road_G.nodes(data=True) if d.get('tile_center') is not None)
print('CSV tiles in bbox:', len(csv_keys))
print('Tiles used by roads:', len(used_keys))
print('Fraction used:', (len(used_keys)/len(csv_keys) if csv_keys else None))