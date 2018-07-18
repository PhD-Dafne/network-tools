import networkx as nx
import collections
import pandas as pd
from collections import Counter
import igraph
import louvain
import numpy as np

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




def consensus_partition(g, 
                        partition_type = louvain.ModularityVertexPartition,
                        weights=None,
                        nr_partitions = 100,
                        threshold = 0,
                        max_nr_iterations = 5,
                       verbose=False):
    '''
    Partitions graph based on consensus clustering
    :param g: igraph Graph
    '''
    n = len(g.vs)
    graph = g
    for j in range(max_nr_iterations):
        if verbose:
            print('Iteration {}'.format(j))

        consensus_matrix = np.zeros((n, n))
        for i in range(nr_partitions):
            partition = louvain.find_partition(graph, partition_type=partition_type, weights=weights)
            k = len(partition.sizes()) # Number of partitions
            b = np.zeros((n, k))
            b[np.arange(n), partition.membership] = 1
            consensus_matrix += b.dot(b.T)
        consensus_matrix /= nr_partitions

        g2 = graph.copy()
        g2.delete_edges(g2.es)

        consensus_matrix_fixed = consensus_matrix.copy()
        consensus_matrix_fixed[consensus_matrix<=threshold] = 0
        ix, jx = consensus_matrix_fixed.nonzero()
        for i,j in zip(list(ix), list(jx)):
            if i!=j: # is this necessary?
                g2.add_edge(i,j,weight=consensus_matrix_fixed[i,j])
        # are there any solo clusters?
        ccs = g2.clusters()
        if verbose:
            print('Smallest connected component: {}'.format(min(ccs.sizes())))

        # plot adjacency matrix
        if verbose:
            plot_sorted_adjacency(consensus_matrix, partition.membership)
        
        # Check if converged
        if(min(consensus_matrix[consensus_matrix.nonzero()])==1):
            if verbose:
                print('Converged!')
            return consensus_matrix, ccs.membership
        graph = g2
        weights = 'weight'
        
    return consensus_matrix, ccs.membership

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