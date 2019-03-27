import igraph
import leidenalg
import numpy as np
import itertools
from nwtools import common, communities


def get_initial_partitions(g, 
                        partition_type = leidenalg.ModularityVertexPartition,
                        weights=None,
                        nr_partitions = 100):
    partitions = []
    for i in range(nr_partitions):
        partitions.append(leidenalg.find_partition(g, partition_type=partition_type, weights=weights))
    return partitions

def get_consensus_matrix(partitions, nr_nodes):
    consensus_matrix = np.zeros((nr_nodes, nr_nodes))
    for partition in partitions:
        k = len(partition.sizes()) # Number of clusters
        b = np.zeros((nr_nodes, k))
        b[np.arange(nr_nodes), partition.membership] = 1
        consensus_matrix += b.dot(b.T)
    consensus_matrix /= len(partitions)
    return consensus_matrix

def consensus_partition(g, initial_partition=None,
                        partition_type = leidenalg.ModularityVertexPartition,
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
        
        # Get the individual partionings
        if j==0 and initial_partition is not None:
            partitions = initial_partition
        else:
            partitions = get_initial_partitions(graph, 
                            partition_type = partition_type,
                            weights=weights,
                            nr_partitions = nr_partitions)
        
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
        if(min(consensus_matrix[consensus_matrix.nonzero()])==1):
            if verbose:
                print('Converged!')
            return consensus_matrix, ccs.membership
        graph = g2
        weights = 'weight'
    return consensus_matrix, ccs.membership



def get_nmi_scores(consensus_membership, all_memberships, average_method='geometric'):
    from sklearn.metrics.cluster import normalized_mutual_info_score
    nmi_consensus = [normalized_mutual_info_score(consensus_membership, mem, average_method=average_method) 
                     for mem in all_memberships]
    nmi_all = [[normalized_mutual_info_score(mem1, mem2, average_method=average_method) 
                for mem1 in all_memberships] for mem2 in all_memberships]
    nmi_all = np.array(nmi_all).flatten()
    return nmi_consensus, nmi_all
    

