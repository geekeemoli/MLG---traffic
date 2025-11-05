# add node attribute pop_density to existing graph
# data structure: longitude,latitude,*_general_2020

import os
import csv
import math
from shapely.geometry import box
from shapely.strtree import STRtree

def get_density(G, csv_path=None):
    """
    Add a node attribute 'pop_density' to graph `G` using population CSV.

    The function expects a CSV where each row contains the center longitude and latitude
    (in degrees) of a 1-arc-second-by-1-arc-second cell and a
    population/density value.

    If a matching cell is found, its value is set on the node as 'pop_density'. If
    no cell matches, 'pop_density' is set to 0.0.

    Parameters
    ----------
    G : networkx.Graph
        Graph whose nodes will be annotated. Node data are modified in-place.
    csv_path : str or None
        Path to population CSV. If None, function will look for
        ../data/population_data/aut_general_2020.csv relative to this file.

    Returns
    -------
    G : networkx.Graph
        The same graph object, with node attributes updated.
    """

    if any('pop_density' in d for _, d in G.nodes(data=True)):
        print("Graph already has 'pop_density' attribute. Stopping function.")
        return G

    for _, data in G.nodes(data=True):
        data['pop_density'] = 0.0

    geoms = []
    geom_id_to_node = {}
    for n, data in G.nodes(data=True):
        geom = data.get("geometry")
        if geom is None:
            print(f'node {n} does not have geometry')
            continue
        geoms.append(geom)
        geom_id_to_node[id(geom)] = n

    if not geoms:
        print("No geometries found on nodes; nothing to index.")
        return G

    tree = STRtree(geoms)

    if csv_path is None:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        csv_path = os.path.join(repo_root, 'data', 'population_data', 'aut_general_2020.csv')

    
    try:
        with open(csv_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            lon_field, lat_field, pop_field = reader.fieldnames[:3]

            for row in reader:
                lon = float(row[lon_field])
                lat = float(row[lat_field])
                val = float(row[pop_field])

                tile = get_tile(lon, lat)
                tile_geom = box(tile["west"], tile["south"], tile["east"], tile["north"])

                candidates = tree.query(tile_geom) # query spatial index → candidate geometries

                for cand in candidates:
                    # map geometry back to node id
                    n = geom_id_to_node[id(cand)]
                    data = G.nodes[n]

                    if cand.intersects(tile_geom):
                        data['pop_density'] = max(data['pop_density'], val) # if the road intersects with more then one tile

    except FileNotFoundError:
        raise FileNotFoundError(
            f"Population CSV not found at {csv_path}; please pass csv_path explicitly"
        )

    return G


def get_tile(lon_center_deg, lat_center_deg):
    half_ddeg = 1.0 / 7200.0

    south = lat_center_deg - half_ddeg
    north = lat_center_deg + half_ddeg
    west = lon_center_deg - half_ddeg
    east = lon_center_deg + half_ddeg

    one_arcsec_lat_m = 30.87
    lat_rad = math.radians(lat_center_deg)
    one_arcsec_lon_m = one_arcsec_lat_m * math.cos(lat_rad)

    return {
        "south": south,
        "north": north,
        "west": west,
        "east": east,
        #if maybe needed:
        "ns_size_m": one_arcsec_lat_m,
        "ew_size_m": one_arcsec_lon_m,
    }


if __name__ == '__main__':
    print('This module provides get_density(G, csv_path=None). It adds atribute pop_density to the graph nodes.')
