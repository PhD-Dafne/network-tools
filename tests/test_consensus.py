import pytest
import numpy as np
import nwtools.consensus

@pytest.fixture
def simple_graph():
    import igraph
    g = igraph.Graph()
    g.add_vertices(5)
    g.add_edges([(0,1), (0,2), (1,2), (2,3), (2,4), (3,4)])
    return g

@pytest.fixture
def simple_partitioning(simple_graph):
    import igraph
    memberships = [i for i in range(simple_graph.vcount())]
    return igraph.clustering.VertexClustering(simple_graph, memberships)

@pytest.fixture
def consensus_matrix():
    nr_nodes = 5
    mat =  np.zeros((nr_nodes, nr_nodes))
    mat[:2, :2] = 1
    mat[2, :] = 0.5
    mat[:, 2] = 0.5
    mat[2,2] = 1
    mat[3:, 3:] = 1
    return mat

@pytest.fixture
def identical_partitionings(simple_partitioning):
    n = 10
    return [simple_partitioning for i in range(n)]

def test_get_initial_partitions(simple_graph):
    nr_partitions = 10
    partitions = nwtools.consensus.get_initial_partitions(simple_graph,
                        nr_partitions = nr_partitions)
    assert len(partitions)==nr_partitions
    

def test_get_consensus_matrix(simple_graph, identical_partitionings):
    partitions = identical_partitionings
    nr_nodes = simple_graph.vcount()
    mat = nwtools.consensus.get_consensus_matrix(partitions, nr_nodes)
    assert mat.shape == (nr_nodes, nr_nodes)
    assert np.all(mat.diagonal()==1)


def test_get_edge_consistency(simple_graph, consensus_matrix):
    graph = nwtools.consensus.get_edge_consistency(simple_graph, consensus_matrix)
    assert len(graph.es['consensus'])==6
    assert np.allclose(graph.es['consensus'], [1,0.5,0.5,0.5,0.5,1])
    assert np.allclose(graph.es['consistency'], [1,0,0,0,0,1])
