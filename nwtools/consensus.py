"""
This module provides different functionalities related to the consensus of
multiple runs of community detection.
The algorithms for community detection are provided by the leidenalg package.
"""
import igraph
import leidenalg
import numpy as np
import scipy.sparse


def get_initial_partitions(g,
                           partition_type=leidenalg.ModularityVertexPartition,
                           weights=None,
                           nr_partitions=100):
    """
    Find initial set of partitions using a community detection method

    :param igraph.Graph g: graph
    :param partition_type: subtype of `leidenalg.VertexPartition`, implementing a community detection algorithm
    :param weights: weight attribute in graph
    :param nr_partitions: number of partitions

    :return: list of partitions
    """
    partitions = []
    for i in range(nr_partitions):
        partitions.append(
            leidenalg.find_partition(g, partition_type=partition_type,
                                     weights=weights))
    return partitions


def get_consensus_matrix(partitions, nr_nodes):
    """
    Get the consensus matrix of a list of partitions.

    :param partitions: iterable of igraph.clustering.VertexClustering
    :param nr_nodes: number of nodes in the graph
    :return: consensus matrix of shape (nr_nodes, nr_nodes)
    :rtype: numpy.array
    """
    consensus_matrix = scipy.sparse.csr_matrix((nr_nodes, nr_nodes))
    for partition in partitions:
        b = scipy.sparse.coo_matrix((np.repeat(1, len(partition.membership)), (
            np.arange(nr_nodes), partition.membership)))
        consensus_matrix += b.dot(b.T)
    consensus_matrix /= len(partitions)
    return consensus_matrix.toarray()


def consensus_partition(g, initial_partition=None,
                        partition_type=leidenalg.ModularityVertexPartition,
                        weights=None,
                        nr_partitions=100,
                        threshold=0,
                        max_nr_iterations=5,
                        singleton_clusters=False,
                        verbose=False):
    """
    Partitions grap based on consensus clustering.
    Reference:
    A.  Lancichinetti,  S.  Fortunato,  Consensus  clustering  in  complex  networks,  Scientific Reports 2(1), 336 (2012).  DOI 10.1038/srep00336

    :param igraph.Graph g: graph
    :param initial_partition: precalculated list of partitions (optional)
    :param leidenalg.VertexPartition partition_type: subtype of community detection algorithm
    :param str weights: weight attribute in graph
    :param int nr_partitions: number of partitions
    :param float threshold: threshold for consensus clustering algorithm
    :param int max_nr_iterations: maximum number of iterations in consensus clustering
    :param bool singleton_clusters: Whether to allow clusters of only one node
    :param bool verbose: If true, print details on progress

    :return: consensus matrix of first iteration, consensus clustering membership of nodes
    :rtype: (np.array, list[int])
    """
    n = len(g.vs)
    graph = g
    first_consensus_matrix = None
    for j in range(max_nr_iterations):
        if verbose:
            print('Iteration {}'.format(j))

        # Get the individual partionings
        if j == 0 and initial_partition is not None:
            partitions = initial_partition
        else:
            partitions = get_initial_partitions(graph,
                                                partition_type=partition_type,
                                                weights=weights,
                                                nr_partitions=nr_partitions)

        # Calculate the consensus matrix
        consensus_matrix = get_consensus_matrix(partitions, n)
        if j == 0:
            first_consensus_matrix = consensus_matrix
        # Create new graph based on consensus matrix
        consensus_matrix_copy = np.triu(consensus_matrix, 1)
        consensus_matrix_copy[consensus_matrix_copy <= threshold] = 0
        # Connect clusters of single nodes to heighest-weight neighbors
        if not singleton_clusters:
            single_nodes = np.where((consensus_matrix_copy.sum(
                axis=1) + consensus_matrix_copy.sum(axis=0)) == 0)[0]
            closest_neighbors = np.argsort(consensus_matrix[single_nodes],
                                           axis=1)[:, -2]
            consensus_matrix_copy[single_nodes, closest_neighbors] = \
                consensus_matrix[single_nodes, closest_neighbors]

        g2 = igraph.Graph.Weighted_Adjacency(consensus_matrix_copy.tolist(),
                                             loops=False, attr='weight',
                                             mode='MAX')

        ccs = g2.clusters()

        # Check if converged
        if (min(consensus_matrix[consensus_matrix.nonzero()]) == 1):
            if verbose:
                print('Converged!')
            return first_consensus_matrix, ccs.membership
        graph = g2
        weights = 'weight'
    return first_consensus_matrix, ccs.membership


def get_nmi_matrix(memberships, average_method='geometric'):
    """
    Calculate full matrix of NMI scores of all combinations of partitions

    :param list[np.array] memberships: list of numpy arrays with node memberships
    :param str average_method: see sklearn.metrics.cluster.normalized_mutual_info_score

    :return: matrix of shape (nr_partitions, nr_partitions) with NMI scores
    :rtype: numpy.array
    """
    from sklearn.metrics.cluster import normalized_mutual_info_score
    nmi_matrix = np.array(
        [[normalized_mutual_info_score(mem1, mem2,
                                       average_method=average_method)
          for mem1 in memberships] for mem2 in memberships])
    return nmi_matrix


def get_nmi_scores(consensus_membership, all_memberships,
                   average_method='geometric'):
    """
    Calculate the NMI score between a consensus partitioning and a list of
    partitionings,  and between all individual partitions of the list.

    :param igraph.clustering.VertexClustering consensus_membership: consensus clustering
    :param all_memberships: iterable of igraph.clustering.VertexClustering
    :param str average_method: see sklearn.metrics.cluster.normalized_mutual_info_score

    :return: (nmi_consensus, nmi_all) - NMI scores
    :rtype: (list, np.array)
    """
    from sklearn.metrics.cluster import normalized_mutual_info_score
    nmi_consensus = [normalized_mutual_info_score(consensus_membership, mem,
                                                  average_method=average_method)
                     for mem in all_memberships]
    nmi_all = get_nmi_matrix(all_memberships, average_method=average_method)
    nmi_all = np.array(nmi_all).flatten()
    return nmi_consensus, nmi_all


def get_unique_partitions(partitions, average_method='geometric'):
    """
    Find identical partitions in a list of partitionings.

    :param partitions: iterable of igraph.clustering.VertexClustering
    :param str average_method: see sklearn.metrics.cluster.normalized_mutual_info_score

    :return: (nmi_all, unique_partitions, unique_part_counts)
    :rtype: (np.array, list[igraph.clustering.VertexClustering], list[int])
    """
    memberships = [p.membership for p in partitions]

    nmi_all = get_nmi_matrix(memberships, average_method=average_method)
    partitions_same = 1 * (nmi_all == 1)
    g_partitions = igraph.Graph.Adjacency(partitions_same.tolist())
    ccs = g_partitions.clusters().membership
    unique_part_index = [np.nonzero(np.array(ccs) == i)[0] for i in set(ccs)]
    unique_part_counts = [len(a) for a in unique_part_index]
    unique_partitions = [partitions[i[0]] for i in unique_part_index]
    return nmi_all, unique_partitions, unique_part_counts


def get_edge_consistency(graph, consensus_matrix):
    """
    Defines consensus and consistency scores for edges in the graph

    :param igraph.Graph graph: igraph Graph
    :param np.array consensus_matrix: the consensus matrix
    :return: the input graph, with edge attributes 'consensus' and 'consistency'
    :rtype: igraph.Graph
    """
    edge_indices = [e.tuple for e in graph.es]
    ix, jx = zip(*edge_indices)
    graph.es['consensus'] = consensus_matrix[ix, jx]
    graph.es['consistency'] = 2 * np.abs(np.array(graph.es['consensus']) - 0.5)
    return graph


def modularity_contribution(graph, membership, weight=None):
    """
    Calculates the gain, penalty and contribution of individual nodes on the
    modularity score, with respect to a clustering
    Based on:
    [1] R. de Santiago and L. C. Lamb,
    “On the role of degree influence in suboptimal modularity maximization,”
    in 2016 IEEE Congress on Evolutionary Computation (CEC),
    Vancouver, BC, Canada, 2016, pp. 4618–4625.

    :param igraph.Graph graph: igraph Graph
    :param list membership: list of cluster membership
    :return: (gain, penalty, contribution)
    :rtype: (np.arrary, np.array, np.array)
    """
    vcount = graph.vcount()
    b = scipy.sparse.coo_matrix((np.repeat(1, len(membership)), (
        np.arange(vcount), membership)))
    c = b.dot(b.T)  # C is the co-clustering matrix
    c = c.toarray()

    if weight is None:
        adj = np.array(graph.get_adjacency().data)
    else:
        adj = np.array(graph.get_adjacency(attribute=weight).data)

    ecount = adj.sum() / 2

    gain = np.sum(adj * c, axis=1) / ecount

    degree = adj.sum(axis=0)  # This could be weighted degree
    d = c * degree
    np.fill_diagonal(d, 0)
    penalty = -np.diag(d.dot(d)) / (2 * ecount ** 2) - degree ** 2 / (
            4 * ecount ** 2)
    contribution = gain - penalty
    return gain, penalty, contribution
