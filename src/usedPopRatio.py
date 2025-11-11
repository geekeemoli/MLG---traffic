import os
import csv
import osmnx as ox
import networkx as nx
from popdensityV3 import *
from shapely.geometry import LineString

#output:
# Processing Graz, Austria
# 11256 roads, 30845 adjacencies
# CSV tiles in bbox: 73758
# Total population in CSV (bbox): 318040.22069084644
#to pa je dejanska uporaba:
# total tile population (used tiles): 26353.222
# sum of assigned node pop_density: 26072.309
# relative difference: -280.913377 (should be ~0) (razlika zaradi najvecje meje razdalje)

# set csv_path to same CSV used by get_density (or pass None to use default)
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
csv_path = os.path.join(repo_root, 'data', 'population_data', 'aut_general_2020.csv')

# Minimal variant of original datacoll.py but using popdensityV3.get_density
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

# compute bbox from your graph G (same as get_density does)
lons = [d.get('x') or d.get('lon') or (d.get('geometry').centroid.x if d.get('geometry') is not None else None) for _,d in G.nodes(data=True)]
lats = [d.get('y') or d.get('lat') or (d.get('geometry').centroid.y if d.get('geometry') is not None else None) for _,d in G.nodes(data=True)]
# filter None
lons = [float(v) for v in lons if v is not None]
lats = [float(v) for v in lats if v is not None]
min_lon, max_lon = min(lons), max(lons)
min_lat, max_lat = min(lats), max(lats)
tile_half = 1.0/7200.0   # same default as code
west = min_lon - tile_half
east = max_lon + tile_half
south = min_lat - tile_half
north = max_lat + tile_half

sum_csv_bbox = 0.0
count_rows = 0
with open(csv_path, newline='', encoding='utf-8') as f:
    r = csv.DictReader(f)
    lon_field, lat_field, pop_field = r.fieldnames[:3]
    for row in r:
        try:
            lon = float(row[lon_field])
            lat = float(row[lat_field])
            val = float(row[pop_field])
        except Exception:
            continue
        if lon < west or lon > east or lat < south or lat > north:
            continue
        sum_csv_bbox += val
        count_rows += 1

print('CSV tiles in bbox:', count_rows)
print('Total population in CSV (bbox):', sum_csv_bbox)