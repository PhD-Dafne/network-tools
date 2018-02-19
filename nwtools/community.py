import networkx as nx
import collections

def partition_statistics(partition, graph, weight=None):
    '''

    :param partition: dict {node: community}
    :param graph:
    :return:
    '''

    reverse_partition = collections.defaultdict(list)
    for n in partition:
        reverse_partition[partition[n]].append(n)

    statistics = {comm: {} for comm in reverse_partition}
    for c in reverse_partition:
        subgraph = graph.subgraph(reverse_partition[c])
        statistics[c]['size'] = nx.number_of_nodes(subgraph)
        statistics[c]['internal_degree'] = nx.number_of_edges(subgraph)
    return statistics