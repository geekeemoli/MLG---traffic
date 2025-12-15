# Processed graphs for Traffic Jam prediction

The graphs are saved like line graphs of the cities with the most important attributes for predicting something like a traffic jam.

### Loading the graph - `data/load_graph.py`
* `load_graph_nx(path)` - returns a Networkx graph of a city at the given path
* `load_graph_pyg(path)` - reutrns a PyG graph of a city at the given path

### Example PyG graph:
```
Data(edge_index=[2, 30845], highway=[11256], lanes=[11256], maxspeed=[11256], length=[11256], pop_density=[11256], correlation=[11256], curvature=[11256], num_nodes=11256)
```

### Each node represent a road segment and every single one contains the following attributes:
* `correlation` (float)- correlation between flow and occupancy of a given road segment (-1 is a full traffic jam, 0 is neutral and 1 is perfect traffic)
* `curvature` (float) - measure of the curvature a road has (0 is no curvature, higher values represent higher curvature)
* `highway` (string) - See [this](#all-highway-parameter-values-and-their-frequencies)
* `lanes` (float) - Average number of lanes along the road
* `length` (float) - Length of the road in meters.
* `maxspeed` (float) - Max speed in km/h
* `pop_density` (float) - Population density on the given road segment


### All `highway` parameter values and their frequencies
| Road Type | Count |
|---|---:|
| residential | 1222938 |
| tertiary | 236910 |
| unclassified | 164360 |
| secondary | 112641 |
| primary | 107108 |
| trunk | 30505 |
| living_street | 27804 |
| motorway_link | 7987 |
| primary_link | 6977 |
| trunk_link | 5787 |
| motorway | 4438 |
| secondary_link | 3953 |
| tertiary_link | 3775 |
| busway | 1328 |
| crossing | 96 |
| road | 34 |
| escape | 13 |
| yes | 8 |
| via_ferrata | 6 |
| alley | 2 |
| emergency_bay | 2 |

A potential way to incorporate this data in a deep learning pipeline is a one-hot encoding for the top k (e.g. 7) categories and adding an additional category 'other' for the rest.

### City graph stats
| City | # nodes | # edges | # traffic-labeled nodes
|---|---:|---:|---:|
| Rotterdam, Netherlands | 25579 | 63466 | 112 |
| Melbourne, Australia | 499153 | 1232527 | 263 |
| Graz, Austria | 11256 | 30845 | 152 |
| Munich, Germany | 36326 | 103316 | 276 |
| Hamburg, Germany | 52871 | 140485 | 206 |
| Paris, France | 18178 | 36811 | 186 |
| Manchester, UK | 35049 | 96181 | 106 |
| Speyer, Germany | 2988 | 8011 | 124 |
| Essen, Germany | 25942 | 68553 | 34 |
| Zurich, Switzerland | 10571 | 27088 | 612 |
| 東京23区 (Tokyo), Japan | 305402 | 880439 | 188 |
| 臺北市 (Taipei), Taiwan | 28441 | 75540 | 399 |
| Los Angeles, USA | 136122 | 423329 | 1480 |
| Marseille, France | 26557 | 61844 | 151 |
| Toulouse, France | 19567 | 45377 | 425 |
| Strasbourg, France | 9364 | 23557 | 122 |
| Torino, Italy | 25242 | 59196 | 360 |
| Kassel, Germany | 9788 | 27153 | 259 |
| Bolton, UK | 27948 | 73806 | 45 |
| Frankfurt, Germany | 20358 | 48043 | 46 |
| London, UK | 303696 | 812652 | 3532 |
| Stuttgart, Germany | 21587 | 58014 | 132 |
| Darmstadt, Germany | 6358 | 17998 | 183 |
| Groningen, Netherlands | 11627 | 29738 | 30 |
| Basel, Switzerland | 4674 | 11830 | 45 |
| Bern, Switzerland | 5882 | 16342 | 326 |
| Cagliari, Italy | 7937 | 17104 | 63 |
| Constance, Germany | 3609 | 9928 | 75 |
| Lucerne, Switzerland | 2613 | 6695 | 81 |
| Madrid, Spain | 61610 | 129851 | 1011 |
| Vilnius, Lithuania | 18534 | 50588 | 303 |
| Augsburg, Germany | 12435 | 35270 | 377 |
| Wolfsburg, Germany | 9052 | 24026 | 71 |
| Bordeaux, France | 8961 | 19484 | 236 |
| Santander, Spain | 103023 | 270685 | 200 |
| Bremen, Germany | 20264 | 52226 | 263 |
| **TOTAL** | | | 12.474 | 