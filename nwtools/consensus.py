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
    adj = np.array(list(graph.get_adjacency(attribute=weights)))
    memberships = []
        
    
        
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
        g2 = graph.copy()
        g2.delete_edges(g2.es)

        ix, jx = np.where(consensus_matrix>0.5)
        for i,j in zip(list(ix), list(jx)):
            if i!=j: 
                g2.add_edge(i,j,weight=consensus_matrix[i,j])
        
        ccs = g2.clusters()
        
        # are there any solo clusters?
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



def get_nmi_scores(consensus_membership, all_memberships, average_method='geometric'):
    nmi_consensus = [normalized_mutual_info_score(consensus_membership, mem, average_method=average_method) 
                     for mem in all_memberships]
    nmi_all = [[normalized_mutual_info_score(mem1, mem2, average_method=average_method) 
                for mem1 in all_memberships] for mem2 in all_memberships]
    nmi_all = np.array(nmi_all).flatten()
    return nmi_consensus, nmi_all
    

