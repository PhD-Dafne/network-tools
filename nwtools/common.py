import networkx as nx
import igraph as ig
import numpy as np

def jaccard(s1, s2):
    s1 = set(s1)
    s2 = set(s2)
    intersect = len(s1.intersection(s2))
    union = len(s1.union(s2))
    if union==0:
        return np.NaN
    return intersect/union

def networkx2igraph(g_nx):
    if(type(g_nx) == nx.DiGraph):
        directed = True
    else:
        directed = False
    g_ig = ig.Graph.TupleList(g.edges(), directed=directed)

    att_list = set(np.array([list(d.keys()) for n, d in g_nx.nodes(data=True)]).flatten())

    for att in att_list:
        att_dict = nx.get_node_attributes(g_nx, att)
        g_ig.vs[att] = [att_dict[n] for n in g_ig.vs['name']]
        
def igraph_from_pandas_edgelist(edges, source='Source', target='Target', attributes=None, directed=False):
    import pandas as pd
    g = ig.Graph(directed=directed)

    node_names = list(set(edges[source]).union(set(edges[target])))

    g.add_vertices(node_names)

    g.add_edges([(s, t) for s, t in edges[[source, target]].values])
    
    for att in attributes:
        g.es[att] = list(edges[att])
        
    return g