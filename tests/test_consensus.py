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
def identical_partitionings(simple_partitioning):
    n = 10
    return [simple_partitioning for i in range(n)]

def test_get_initial_partitions(simple_graph):
    partitions = nwtools.consensus.get_initial_partitions(simple_graph, 
                        nr_partitions = 10)
    assert len(partitions)==10
    

def test_get_consensus_matrix(simple_graph, identical_partitionings):
    partitions = identical_partitionings
    nr_nodes = simple_graph.vcount()
    mat = nwtools.consensus.get_consensus_matrix(partitions, nr_nodes)
    assert mat.shape == (nr_nodes, nr_nodes)
    assert np.all(mat.diagonal()==1)