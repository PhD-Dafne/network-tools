import networkx as nx
import collections
import pandas as pd
from collections import Counter
import numpy as np
import itertools
from nwtools import common

def partition_statistics(partition, graph, weight=None):
    '''
    Calculates statistics for each partition of a network.
    Treats the network as undirected.
    
    :param partition: dict {node: community}
    :param graph: networkx graph
    :return: statistics
    '''

    reverse_partition = collections.defaultdict(list)
    for n in partition:
        reverse_partition[partition[n]].append(n)

    statistics = {comm: {} for comm in reverse_partition}
    for c in reverse_partition:
        subgraph = graph.subgraph(reverse_partition[c])
        statistics[c]['size'] = nx.number_of_nodes(subgraph)
        statistics[c]['total_degree'] = sum(dict(graph.degree(reverse_partition[c])).values())
        statistics[c]['average_degree'] = statistics[c]['total_degree'] / statistics[c]['size']
        statistics[c]['internal_degree'] = nx.number_of_edges(subgraph)
        statistics[c]['average_internal_degree'] = statistics[c]['internal_degree'] / statistics[c]['size']
        statistics[c]['external_degree'] = statistics[c]['total_degree'] - statistics[c]['internal_degree']
        statistics[c]['average_external_degree'] = statistics[c]['external_degree'] / statistics[c]['size']
        statistics[c]['conductance'] = statistics[c]['external_degree'] / statistics[c]['total_degree']
        if weight is not None:
            statistics[c]['total_strength'] = sum(dict(graph.degree(reverse_partition[c], weight)).values())
            statistics[c]['average_strength'] = statistics[c]['total_strength'] / statistics[c]['size']
            statistics[c]['internal_strength'] = sum(nx.get_edge_attributes(subgraph, weight).values())
            statistics[c]['average_internal_strength'] = statistics[c]['internal_strength'] / statistics[c]['size']
            statistics[c]['external_strength'] = statistics[c]['total_strength'] - statistics[c]['internal_strength']
            statistics[c]['average_external_strength'] = statistics[c]['external_strength'] / statistics[c]['size']
            statistics[c]['weighted_conductance'] = statistics[c]['external_strength'] / statistics[c]['total_strength']
    return statistics


def compare_communities(partition1, partition2, graph):
    '''
    Calculates different similarity measures between two partitions of a graph
    '''
    result = {}
    partition_merged = pd.DataFrame({'p1': partition1, 'p2':partition2})
    result['nodes_contigency_table'] = partition_merged.pivot_table(index='p1', columns='p2', aggfunc=len, fill_value=0)
    
    # Looks at pairs
    pairs_p1 = partition_merged.reset_index().merge(partition_merged.reset_index(), on='p1')
    pairs_p1 = pairs_p1[pairs_p1['index_x']!=pairs_p1['index_y']]
    pairs_p2 = partition_merged.reset_index().merge(partition_merged.reset_index(), on='p2')
    pairs_p2 = pairs_p2[pairs_p2['index_x']!=pairs_p2['index_y']]
    pairs_p1p2 = pairs_p1.merge(pairs_p2, on=['index_x', 'index_y'])
    
    a11 = len(pairs_p1p2)
    a10 = len(pairs_p1) - a11
    a01 = len(pairs_p2) - a11
    a00 = len(partition_merged)*(len(partition_merged)-1) - a11 - a10 - a01
    
    result['rand_index'] = (a11+a00)/(a11+a01+a10+a00)
    result['jaccard_index'] = a11/(a11+a01+a10)
    
    # Information theoretical measures, see https://bitbucket.org/dsign/gecmi/wiki/Home
    n = len(partition_merged)
    p_xy = partition_merged.groupby(['p1', 'p2']).apply(len)/n
    return result



def plot_sorted_adjacency(adj, membership):
    import matplotlib.pyplot as plt
    order = np.argsort(membership)
    adj_sorted = adj[order][:,order]
    ax = plt.imshow(adj_sorted, cmap='Greys', interpolation='none')

    n = adj.shape[0]
    cluster_sizes = Counter(membership)
    cluster_sizes = [cluster_sizes[c] for c in set(sorted(membership))]
    cumsums = np.concatenate(([0], np.cumsum(cluster_sizes)))
    for x, y in zip(cumsums[:-1],cumsums[1:]):
        plt.hlines(x-.5, x-.5, y-.5, color='red', antialiased=False)
        plt.vlines(x-.5, x-.5, y-.5, color='red', antialiased=False)
        plt.hlines(y-.5, x-.5, y-.5, color='red', antialiased=False)
        plt.vlines(y-.5, x-.5, y-.5, color='red', antialiased=False)

    plt.xlim(0, n-.5)
    plt.ylim(n-.5, 0)
    plt.show()
    

def jaccard_crosstab(part1, part2, keys1, keys2):
    sim = np.zeros((len(part1), len(part2)))
    for (i, c1), (j, c2) in itertools.product(enumerate(keys1), enumerate(keys2)):
        sim[i, j] = common.jaccard(part1[c1], part2[c2])
    return sim


def hungarian_algorithm(ctab, row_labels, col_labels):
    from scipy.optimize import linear_sum_assignment
    if ctab.shape[0] > ctab.shape[1]:
        raise Exception('Rows should be fewer than columns')
        
    # Use the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(- ctab)
    mapXtoY = {row_labels[r]: col_labels[c] for r, c in zip(row_ind, col_ind)}
    return(mapXtoY)

def map_labels_over_time(labels_list, jaccard=True, min_overlap=0.1, character_labels=True):
    if character_labels:
        import string
        new_labels = list(string.ascii_lowercase + string.ascii_uppercase)
    else:
        max_nr_of_labels = np.sum([len(s) for s in labels_list])
        new_labels = list(range(max_nr_of_labels))
    max_new_label = 0
    
    mappings = []

    part1 = None
    for t in range(len(labels_list)):
        part2 = labels_list[t]
        if part1 is None:
            m = {l: new_labels[i] for (i,l) in enumerate(part2.keys())}
            max_new_label = len(part2)
        else:
            col_labels = sorted(part1.keys()) 
            row_labels = sorted(part2.keys()) 

            ctab = jaccard_crosstab(part2, part1, row_labels, col_labels)

            # Add dummy columns
            ctab2 = np.hstack((ctab, np.ones((len(part2), len(part2)))*min_overlap))
            col_labels = col_labels + new_labels[max_new_label:max_new_label+len(part2)]
            m = hungarian_algorithm(ctab2, row_labels, col_labels)
            max_new_label = new_labels.index(max(m.values()))+1
        mappings.append(m)
        part1 = {m[c]: part2[c] for c in part2}
    return mappings

def citation_distance_matrix(graph):
    '''
    :param graph: networkx graph
    returns: distance matrix, node labels
    '''
    sinks = [key for key, outdegree in graph.out_degree() if outdegree==0]
    paths = {s: nx.shortest_path_length(graph, target=s) for s in sinks}
    paths_df = pd.DataFrame(paths)#, index=graph.nodes)
    paths_nonzero_df = 1*~paths_df.isnull()
    a_paths_nonzero = paths_nonzero_df.values
    m = a_paths_nonzero
    intersect = m.dot(m.T)
    union = m.dot(np.ones(m.shape).T) + np.ones(m.shape).dot(m.T) -intersect
    union[union==0] = 1
    dist = 1 - intersect/union
    return dist, paths_nonzero_df.index