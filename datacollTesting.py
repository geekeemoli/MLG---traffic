import os
import csv
import osmnx as ox
import networkx as nx
from src import popdensityV5 as p
from shapely.geometry import LineString

# Base directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
POP_DATA_DIR = os.path.join(SCRIPT_DIR, "data", "population_data")

# Country name to CSV file prefix mapping
COUNTRY_TO_CSV_PREFIX = {
    "Austria": "aut",
    "Germany": "deu",
    "Switzerland": "che",
    "UK": "gbr",
    "France": "fra",
    "Italy": "ita",
    "Netherlands": "nld",
    "Spain": "esp",
    "USA": "usa",
    "Australia": "aus",
    "Taiwan": "twn",
    "Japan": "jpn",
    "Lithuania": "ltu",
}

def get_csv_path_for_city(city_str):
    """
    Get the population CSV path for a city string like "Graz, Austria".
    Searches for any CSV file in POP_DATA_DIR that contains the country prefix.
    Returns the path if a matching CSV exists, otherwise None.
    """
    parts = city_str.split(",")
    if len(parts) < 2:
        print(f"Warning: Cannot parse country from '{city_str}'")
        return None
    
    country = parts[-1].strip()
    prefix = COUNTRY_TO_CSV_PREFIX.get(country)
    
    if prefix is None:
        print(f"Warning: No CSV prefix mapping for country '{country}'")
        return None
    
    try:
        files = os.listdir(POP_DATA_DIR)
    except FileNotFoundError:
        print(f"Warning: Population data directory not found: {POP_DATA_DIR}")
        return None
    
    # Search for CSV files containing the country prefix anywhere in the filename
    matching_csvs = [f for f in files if prefix in f.lower() and f.endswith('.csv')]
    
    if not matching_csvs:
        print(f"Warning: No CSV file found containing '{prefix}' in {POP_DATA_DIR}")
        return None
    
    if len(matching_csvs) > 1:
        print(f"Warning: Multiple CSV files found containing '{prefix}': {matching_csvs}. Using first one.")
    
    csv_path = os.path.join(POP_DATA_DIR, matching_csvs[0])
    return csv_path

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
    
    # Get the appropriate CSV path for this city's country
    csv_path = get_csv_path_for_city(city)
    if csv_path is None:
        print(f"Skipping {city} - no population data available")
        continue
    
    print(f"Using population data: {csv_path}")
    graph_popd = p.get_density(graphs[city], csv_path, verbose=True)
    # quick samples
    # print('sample with pop_density>0:', [n for n,d in graph_popd.nodes(data=True) if d.get('pop_density',0)>0][:8])
    # summary = p4.analyze_density(graph_popd, top_n=100, verbose=True)

    # # export reports: far-assigned nodes and unassigned nodes
    # reports_dir = os.path.join(os.path.dirname(__file__), 'analysis_reports')
    # os.makedirs(reports_dir, exist_ok=True)

    # # Export top roads by tiles and top tiles by roads GeoJSONs
    # p4.export_top_roads_tiles_geojson(graph_popd, reports_dir, top_n=100, city_name=city)