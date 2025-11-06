# add node attribute pop_density to existing graph
# data structure: longitude,latitude,*_general_2020

import os
import csv
import math
import numbers
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
        data['pop_density'] = 0
        data['prvic'] = 1

    geoms = []
    # map tree index -> node id
    index_to_node = []
    for n, data in G.nodes(data=True):
        geom = data.get("geometry")
        if geom is None:
            print(f'node {n} does not have geometry')
            continue
        geoms.append(geom)
        index_to_node.append(n)

    if not geoms:
        print("No geometries found on nodes; nothing to index.")
        return G

    tree = STRtree(geoms)

    # map geometry id -> index for Shapely versions that return geometry objects from query
    geom_id_to_index = {id(g): idx for idx, g in enumerate(geoms)}

    if csv_path is None:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        csv_path = os.path.join(repo_root, 'data', 'population_data', 'aut_general_2020.csv')

    
    try:
        with open(csv_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            lon_field, lat_field, pop_field = reader.fieldnames[:3]
            # total_lines = sum(1 for _ in csvfile)
            # print(f'total lines in {csv_path}: {total_lines}')
            # stevilo vrstic: 17034412 (aut_general_2020)

            skupno = 0
            unikatnih = 0
            
            i=0
            for row in reader:
                if i%100000 == 0:
                    print(f'we are on index: {i}')
                i += 1
                
                lon = float(row[lon_field])
                lat = float(row[lat_field])
                val = float(row[pop_field])

                tile = get_tile(lon, lat)
                tile_geom = box(tile["west"], tile["south"], tile["east"], tile["north"])

                # get list of indices of candidate geometries from the spatial index
                # Some Shapely versions provide query_items (returns indices), others provide query (returns geometries or indices).
                if hasattr(tree, "query_items"):
                    candidate_indices = list(tree.query_items(tile_geom))
                else:
                    candidates = tree.query(tile_geom)
                    # map geometries or numeric indices back to indices (robust across shapely versions)
                    candidate_indices = []
                    for cand in candidates:
                        # cand may be a geometry object or an integer index (numpy.int64)
                        if isinstance(cand, numbers.Integral):
                            candidate_indices.append(int(cand))
                        else:
                            idx = geom_id_to_index.get(id(cand))
                            if idx is not None:
                                candidate_indices.append(idx)
                             
                for idx in candidate_indices:
                    n = index_to_node[idx]
                    cand = geoms[idx]
                    data = G.nodes[n]

                    if cand.intersects(tile_geom):
                        skupno += 1
                        if  data['prvic'] == 1:
                            unikatnih += 1
                            data['prvic'] = 0
                        
                        data['pop_density'] = max(data['pop_density'], val) # if the road intersects with more then one tile
                            
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Population CSV not found at {csv_path}; please pass csv_path explicitly"
        )

    print(f'skupno zadetkov: {skupno}')
    print(f'unikatnih zadetkov: {unikatnih}')
    
    nezadetih = 0
    for n, data in G.nodes(data=True):
        if data.get('prvic') == 1:
            if nezadetih % 50 == 0:
                print(data.get('geometry'))
            nezadetih += 1
    
    print(f'dejansko nismo zadeli {nezadetih} cest')
    
    return G


def get_tile(lon_center_deg, lat_center_deg):
    #half_ddeg = 1.0 / 7200.0 #nezadetih: 824 (NS_m=30.92m  EW_m=21.48m)
    #half_ddeg = 1.0 / 6000.0 #nezadetih: 599 (NS_m=37.11m  EW_m=25.78m => NS dodatnih 3.5m, 2m)
    half_ddeg = 1.0 / 5000.0 #nezadetih: 413 (NS_m=44.53m  EW_m=30.93m => NS dodatnih 7m, EW dodatne 4.5m)

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
