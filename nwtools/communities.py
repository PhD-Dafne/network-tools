import networkx as nx
import collections
import pandas as pd

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