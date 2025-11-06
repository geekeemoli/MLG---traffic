import osmnx as ox
import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict

def map_detectors_to_road_graph(
    detector_coords_file: str = "detectors_public.csv",
    cities: List[str] = ["Graz, Austria", "Munich, Germany"],
    detector_ids: List[str] = None
) -> Dict[str, nx.Graph]:

    # Load detector coordinates
    detectors_df = pd.read_csv(detector_coords_file)

    #city mapping
    city_mapping = {
        "Graz, Austria": "graz",
        "Munich, Germany": "munich"
    }

    graphs = {}

    for city in cities:
        print(f"\n{'='*60}")
        print(f"Processing {city}")
        print(f"{'='*60}")

        # Get city code (it is more city name) for filtering detectors -> to oobtain "graz", as this is how they are represented in detectors_public.csv
        city_code = city_mapping.get(city, city.split(',')[0].lower())

        # Filter detectors for this city, select only those for specified cities
        city_detectors = detectors_df[
            detectors_df['citycode'].str.lower() == city_code
        ].copy()

        #a warning sign, if no detectors for the city found
        if len(city_detectors) == 0:
            print(f"Warning: No detectors found for {city}")
            continue

        #if detectors are found, print out the number of them for the corresponding city
        print(f"Found {len(city_detectors)} detectors in {city}")

        #Download street network and print num of nodes and vertices
        G = ox.graph_from_place(city, network_type="drive")
        print(f"Original graph: {len(G.nodes)} nodes, {len(G.edges)} edges")
        

        #Initialize detector attributes for all road segments, on edges in the original graph
        #for edge in G.edges(keys=True):
            #G.edges[edge]['has_detector'] = False
            #G.edges[edge]['detectors'] = []
        
        # Extract detector coordinates
        detector_long = city_detectors['long'].values
        detector_lat = city_detectors['lat'].values
        detector_ids = city_detectors['detid'].values

        #find nearest edge in the original graph
        nearest_edges = ox.distance.nearest_edges(G, detector_long, detector_lat, return_dist=False)
        mapping = dict(zip(detector_ids, nearest_edges))

        G_roads = nx.line_graph(G)
        print(f"Line graph: {len(G_roads.nodes)} nodes, {len(G_roads.edges)} edges")

        # Copy edge attributes from original graph to line graph nodes
        #check if this is neccesary or is this automatically done with nx-line_graph()
        for u, v, k, data in G.edges(keys=True, data=True):
            lg_node = (u, v, k)
            if lg_node in G_roads:
                G_roads.nodes[lg_node].update(data)
        
        detectors_added = 0
        
        for detid, edge in mapping.items():
            if edge in G_roads.nodes:
                # Initialize detector list if not exists
                if 'detectors' not in G_roads.nodes[edge]:
                    G_roads.nodes[edge]['detectors'] = []
                    G_roads.nodes[edge]['has_detector'] = True
                
                # Add detector ID to this road segment
                G_roads.nodes[edge]['detectors'].append(detid)
                detectors_added += 1
        
        for node in G_roads.nodes:
            if 'has_detector' not in G_roads.nodes[node]:
                G_roads.nodes[node]['has_detector'] = False
                G_roads.nodes[node]['detectors'] = []
        
        graphs[city] = G_roads

        # Print sample of mapped detectors
        print("\nSample of detector mappings:")
        sample_nodes = [n for n in G_roads.nodes if G_roads.nodes[n]['has_detector']][:3]
        for node in sample_nodes:
            attrs = G_roads.nodes[node]
            print(f"  Road segment {node}:")
            print(f"    Detectors: {attrs['detectors']}")
            print(f"    Road name: {attrs.get('name', 'Unknown')}")
            print(f"    Length: {attrs.get('length', 'Unknown')} m")
    
    return graphs


#main part of the py script
if __name__ == "__main__":
    print("--- Starting detector mapping script ---")

    try:
        # Call your function to get the graphs
        # This uses the default cities: ["Graz, Austria", "Munich, Germany"]
        city_graphs = map_detectors_to_road_graph()
        
        # --- Basic Inspection ---
        for city_name, graph in city_graphs.items():
            print(f"\n--- Inspecting results for: {city_name} ---")
            
            # Check that it's a real graph
            if graph and isinstance(graph, nx.Graph):
                print(f"Graph created with {len(graph.nodes)} road segments (nodes).")
                
                # Check how many nodes have detectors
                nodes_with_detectors = [
                    n for n, data in graph.nodes(data=True) 
                    if data.get('has_detector', False)
                ]
                
                print(f"Found {len(nodes_with_detectors)} road segments with detectors.")
            
            else:
                print(f"Error: No valid graph was returned for {city_name}.")

    except Exception as e:
        print(f"\n--- AN UNEXPECTED ERROR OCCURRED ---")
        print(e)












        








