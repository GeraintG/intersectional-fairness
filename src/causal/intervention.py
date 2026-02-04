import numpy as np

def do_intervention(graph, node, value):
    """
    Perform a do-intervention on a node in the causal graph.
    
    Args:
        graph: Causal graph object
        node: Node to intervene on
        value: Value to set the node to
        
    Returns:
        Modified adjacency matrix after intervention
    """
    adjacency_matrix = graph.get_adjacency_matrix()
    node_index = graph.graph.nodes().index(node)
    
    # Set all incoming edges to the node to 0
    adjacency_matrix[:, node_index] = 0
    
    # Set the node's value
    adjacency_matrix[node_index, node_index] = value
    
    return adjacency_matrix

def truncate_causal_path(graph, source, target):
    """
    Truncate the causal path between two nodes by removing edges.
    
    Args:
        graph: Causal graph object
        source: Source node
        target: Target node
        
    Returns:
        Modified adjacency matrix after truncation
    """
    adjacency_matrix = graph.get_adjacency_matrix()
    source_index = graph.graph.nodes().index(source)
    target_index = graph.graph.nodes().index(target)
    
    # Remove all edges along the path from source to target
    path = nx.shortest_path(graph.graph, source, target)
    for i in range(len(path) - 1):
        u_index = graph.graph.nodes().index(path[i])
        v_index = graph.graph.nodes().index(path[i+1])
        adjacency_matrix[u_index, v_index] = 0
    
    return adjacency_matrix