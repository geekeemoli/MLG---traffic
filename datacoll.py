import osmnx as ox
import networkx as nx
from src.popdensityV2 import get_density
from shapely.geometry import Point, LineString

#, "Munich, Germany"
cities = ["Graz, Austria"]
graphs = {}


for city in cities:
    print(f"Processing {city}")
    G = ox.graph_from_place(city, network_type="drive")

    # dodamo geometry unim ki se manjka (baje OSMnx ne da geometry ravnim crtam)
    for u, v, k, data in G.edges(keys=True, data=True):
        if 'geometry' not in data:
            x1 = G.nodes[u].get('x'); y1 = G.nodes[u].get('y')
            x2 = G.nodes[v].get('x'); y2 = G.nodes[v].get('y')
            if None not in (x1, y1, x2, y2):
                data['geometry'] = LineString([(x1, y1), (x2, y2)])

    road_G = nx.line_graph(G)
    # print(f'avstrija ima toliko vozlisc: {road_G.number_of_nodes()}')
    #avstrija ima toliko vozlisc: 11256
    
    # --- copy edge attributes from G to line-graph node attributes ---
    for u, v, k, data in G.edges(keys=True, data=True):
        lg_node = (u, v, k)              
        if lg_node in road_G:
            road_G.nodes[lg_node].update(data)


    graphs[city] = road_G
    print(f"{len(road_G.nodes)} roads, {len(road_G.edges)} adjacencies")

#    node_ids = list(road_G.nodes)
#    for sample_node in node_ids[:3]:
#        tenth_node_attrs = road_G.nodes(data=True)[sample_node]  # 10th by iteration order
#        print(sample_node, " node attrs: ", tenth_node_attrs)

for city in cities:
    print(f'adding density to {city}')
    graph_popd = get_density(graphs[city])

    # total = graph_popd.number_of_nodes()
    # with_geom = sum(1 for _,d in graph_popd.nodes(data=True) if d.get('geometry') is not None)
    # has_attr = sum(1 for _,d in graph_popd.nodes(data=True) if 'pop_density' in d)
    # pop_gt0 = sum(1 for _,d in graph_popd.nodes(data=True) if d.get('pop_density', 0.0) > 0.0)
    # prviced = sum(1 for _,d in graph_popd.nodes(data=True) if d.get('prvic', 1) == 0)

    # print("total nodes:", total)
    # print("nodes with geometry:", with_geom)
    # print("nodes with pop_density attribute:", has_attr)
    # print("nodes with pop_density > 0:", pop_gt0)
    # print("nodes with unique hits:", prviced)

    # print("sample nodes with pop_density>0:", [n for n,d in graph_popd.nodes(data=True) if d.get('pop_density',0)>0][:8])
    # print("sample nodes with pop_density==0 (but have geometry):", [n for n,d in graph_popd.nodes(data=True) if d.get('geometry') is not None and d.get('pop_density',0)==0][:8])