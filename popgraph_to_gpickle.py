import os
import csv
import osmnx as ox
import networkx as nx
from src import popdensityV5 as p
from shapely.geometry import LineString
import time
import pickle
import sys

# Base directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
POP_DATA_DIR = os.path.join(SCRIPT_DIR, "data", "population_data")
LOG_FILE = os.path.join(SCRIPT_DIR, "datacoll_log.txt")

# Custom class to write to both terminal and file
class Logger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "a", encoding="utf-8")
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Redirect stdout to both terminal and log file
sys.stdout = Logger(LOG_FILE)
print(f"\n{'='*60}")
print(f"Run started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*60}")

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

cities = [
    "Augsburg, Germany",
    "Basel, Switzerland",
    "Berne, Switzerland",
    "Birmingham, UK",
    "Bolton, UK",
    "Bremen, Germany",
    "Bordeaux, France",
    "Cagliari, Italy",
    "Constance, Germany",
    "Darmstadt, Germany",
    "Essen, Germany",
    "Frankfurt, Germany",
    "Graz, Austria",
    "Groningen, Netherlands",
    "Hamburg, Germany",
    "Innsbruck, Austria",
    "Kassel, Germany",
    "London, UK",
    "Los Angeles, USA",
    "Lucerne, Switzerland",
    "Madrid, Spain",
    "Melbourne, Australia",
    "Manchester, UK",
    "Marseille, France",
    "Munich, Germany",
    "Paris, France",
    "Rotterdam, Netherlands",
    "Santander, Spain",
    "Speyer, Germany",
    "Strasbourg, France",
    "Stuttgart, Germany",
    "臺北市, Taiwan",  # Taipei City
    "東京23区, Japan", # Tokyo 23 Wards
    "Torino, Italy",
    "Toulouse, France",
    "Utrecht, Netherlands",
    "Vilnius, Lithuania",
    "Wolfsburg, Germany",
    "Zurich, Switzerland"
]

# gdf = ox.geocode_to_gdf("Tokyo, Japan")
# print(gdf[['display_name', 'type', 'class']])
# print(f"Bounds: {gdf.total_bounds}")

for city in cities:
    output_path = os.path.join(SCRIPT_DIR, 'data', 'graphs', f"{city.replace(', ', '_').replace(' ', '_')}_road_graph_with_popdensity.gpickle")
    if os.path.exists(output_path):
        print(f"Graph with population density for {city} already exists at {output_path}, skipping...\n")
        continue

    start_time = time.time()
    print(f"Processing {city}")
    G = ox.graph_from_place(city, network_type="drive")

    for u, v, k, data in G.edges(keys=True, data=True):
        if 'geometry' not in data:
            x1 = G.nodes[u].get('x'); y1 = G.nodes[u].get('y')
            x2 = G.nodes[v].get('x'); y2 = G.nodes[v].get('y')
            if None not in (x1, y1, x2, y2):
                data['geometry'] = LineString([(x1, y1), (x2, y2)])

    road_G = nx.line_graph(G)

    for u, v, k, data in G.edges(keys=True, data=True):
        lg_node = (u, v, k)
        if lg_node in road_G:
            road_G.nodes[lg_node].update(data)

    print(f"{len(road_G.nodes)} roads, {len(road_G.edges)} adjacencies")

    csv_path = get_csv_path_for_city(city)
    if csv_path is None:
        print(f"Skipping {city} - no population data available")
        continue
    
    print(f"Using population data: {csv_path}")
    graph_popd = p.get_density(road_G, csv_path, verbose=True)

    output_path = os.path.join(SCRIPT_DIR, 'data', 'graphs', f"{city.replace(', ', '_').replace(' ', '_')}_road_graph_with_popdensity.gpickle")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(graph_popd, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved graph with population density to {output_path}")
    print(f"Finished processing {city} in {time.time() - start_time:.2f} seconds\n")
