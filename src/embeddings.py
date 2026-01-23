import networkx as nx
import numpy as np
from node2vec import Node2Vec

def generate_embeddings(adj_matrix, dim=512):
    """
    Generates structural embeddings using Node2Vec.
    Robust replacement for LINE.
    """
    print(" -> Generating embeddings (this may take a moment)...")
    G = nx.from_numpy_array(adj_matrix)
    
    # Fast settings for Node2Vec
    n2v = Node2Vec(G, dimensions=dim, walk_length=30, num_walks=200, workers=4, quiet=True)
    model = n2v.fit(window=10, min_count=1, batch_words=4)
    
    # Ensure correct ordering
    embeddings = np.array([model.wv[str(i)] for i in range(len(G.nodes()))])
    return embeddings