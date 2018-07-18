import networkx as nx
import igraph as ig

def networkx2igraph(g_nx):
    g_ig = ig.Graph.TupleList(g.edges())
    # TODO: is it a directed graph?
    att_list = set(np.array([list(d.keys()) for n, d in g_nx.nodes(data=True)]).flatten())

    for att in att_list:
        att_dict = nx.get_node_attributes(g_nx, att)
        g_ig.vs[att] = [att_dict[n] for n in g_ig.vs['name']]