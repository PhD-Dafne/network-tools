import igraph
import leidenalg
import numpy as np
import scipy.sparse


def get_initial_partitions(g,
                           partition_type=leidenalg.ModularityVertexPartition,
                           weights=None,
                           nr_partitions=100):
    partitions = []
    for i in range(nr_partitions):
        partitions.append(
            leidenalg.find_partition(g, partition_type=partition_type,
                                     weights=weights))
    return partitions


def get_consensus_matrix(partitions, nr_nodes):
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

        # Create new graph based on consensus matrix
        consensus_matrix_copy = np.triu(consensus_matrix, 1)
        consensus_matrix_copy[consensus_matrix_copy <= threshold] = 0
        g2 = igraph.Graph.Weighted_Adjacency(consensus_matrix_copy.tolist(),
                                             loops=False, attr='weight',
                                             mode='MAX')

        ccs = g2.clusters()

        # are there any solo clusters?
        if verbose:
            print('Smallest connected component: {}'.format(min(ccs.sizes())))

        # Check if converged
        if (min(consensus_matrix[consensus_matrix.nonzero()]) == 1):
            if verbose:
                print('Converged!')
            return consensus_matrix, ccs.membership
        graph = g2
        weights = 'weight'
    return consensus_matrix, ccs.membership


def get_nmi_scores(consensus_membership, all_memberships,
                   average_method='geometric'):
    from sklearn.metrics.cluster import normalized_mutual_info_score
    nmi_consensus = [normalized_mutual_info_score(consensus_membership, mem,
                                                  average_method=average_method)
                     for mem in all_memberships]
    nmi_all = [[normalized_mutual_info_score(mem1, mem2,
                                             average_method=average_method)
                for mem1 in all_memberships] for mem2 in all_memberships]
    nmi_all = np.array(nmi_all).flatten()
    return nmi_consensus, nmi_all


def get_edge_consistency(graph, consensus_matrix):
    """
    Defines consensus and consistency scores for edges in the graph

    :param graph: igraph object
    :param consensus_matrix: numpy matrix with the consensus matrix
    :return: the input graph, with edge attributes 'consensus' and 'consistency'
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

    :param graph: igraph object
    :param membership: list of cluster membership
    :return:
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