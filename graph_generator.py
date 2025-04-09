import utils.utils as utils
from sklearn.neighbors import NearestNeighbors
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from igraph import Graph
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix
import torch

def mst_graph(X, metric):

    if metric == 'cosine':
        D = cosine_distances(X, X)
    else:
        D = euclidean_distances(X, X)
    adj_directed = minimum_spanning_tree(D).toarray()
    adj = adj_directed + adj_directed.T
    adj[adj > 0] = 1
    np.fill_diagonal(adj,0)

    return csr_matrix(adj)
    
class Predictor(object):

    def __init__(self, W, h=2, eligible=None, excluded=None):

        sources, targets = W.nonzero()
        edgelist = zip(sources.tolist(), targets.tolist())
        weights = np.array(W[sources, targets])[0]
        self.G = Graph(W.shape[0], edgelist, edge_attrs={"weight": weights})
        self.W = W
        self.adjlist = list(map(set, self.G.get_adjlist()))
        self.eligible_attr = eligible
        self.excluded = [] if excluded is None else excluded
        self.h = h

    def predict(self, *args, **kwargs):
        raise NotImplementedError

    def add_edges(self):
        raise NotImplementedError

    def neighbourhood(self, v):
        """Get k-neighbourhood of node n"""

        if self.h == 1:
            return self.adjlist[v]
        neighbors = self.G.neighborhood(vertices=v, order=self.h)
        return neighbors

    def eligible(self, u, v):
        """Check if link between nodes u and v is eligible
        Eligibility allows us to ignore some nodes/links for link prediction.
        """

        return self.eligible_node(u) and self.eligible_node(v) and u != v

    def eligible_node(self, v):
        """Check if node v is eligible
        Eligibility allows us to ignore some nodes/links for link prediction.
        """

        if self.eligible_attr is None:
            return True
        return self.G.vs[v][self.eligible_attr]

    def eligible_nodes(self):
        """Get list of eligible nodes
        Eligibility allows us to ignore some nodes/links for link prediction.
        """

        return [v for v in self.G if self.eligible_node(v)]

    def likely_pairs(self):
        """
        Yield node pairs from the same neighbourhood
        Arguments
        ---------
        k : int
            size of the neighbourhood (e.g., if k = 2, the neighbourhood
            consists of all nodes that are two links away)
        """

        for a in range(0, self.G.vcount()):
            if not self.eligible_node(a):
                continue
            for b in self.neighbourhood(a):
                if not self.eligible_node(b):
                    continue
                yield (a, b)


class WeightedCommonNeighbours(Predictor):

    def __init__(self, *args, **kwargs):
        self.top_links = kwargs.pop('top_links')
        self.links = None
        super().__init__(*args, **kwargs)

    def predict(self):

        self.links = dict()
        for i, j in self.likely_pairs():
            if self.G[i, j] or i == j:
                continue
            _sum = 0.0
            cn = self.adjlist[i].intersection(self.adjlist[j])
            for z in cn:
                _sum += (self.G[i, z] + self.G[j, z]) / 2
            flag = self.links.get((i, j), False)
            if i < j and not flag:
                self.links[(i, j)] = _sum
            elif not flag:
                self.links[(j, i)] = _sum
        return self.links

    def add_edges(self):

        self.top_links = self.W.count_nonzero() * self.top_links
        for itr, vertices in enumerate(sorted(self.links, key=self.links.get, reverse=True)):
            i, j = vertices
            if itr > self.top_links:
                break
            self.W[i, j] = self.links[(i, j)]

        return self.W

def graph_generator(X, graph_type: str, k = None):
    if graph_type == "KNN":
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(X)
        distances, indices = nbrs.kneighbors(X)
        edge_index = []
        for i in range(X.shape[0]):
            for j in indices[i, 1:]:  # ignora o próprio ponto
                edge_index.append([i, j])

        # Converte para array numpy e ajusta a forma para [2, num_edges]
        edge_index = np.array(edge_index).T
        return edge_index
    
    if graph_type == "random":
        edge_index = np.random.uniform(2,3, 10)
        return edge_index
        
    if graph_type == "MST":
        # Calcula a matriz de distâncias euclidianas entre os pontos
        dist_matrix = squareform(pdist(X, metric='euclidean'))

        # Computa a MST usando Prim (implementado no scipy)
        mst = minimum_spanning_tree(dist_matrix)

        # Obtém as arestas e os vértices da MST
        mst_coo = mst.tocoo()  # Converte para formato COO para acessar os índices

        # Monta o edge_index no formato [2, num_edges]
        edge_index = np.vstack((mst_coo.row, mst_coo.col))
        return edge_index
        
    if graph_type == "GBLP":
        W = mst_graph(X, 'euclidean')
        model = WeightedCommonNeighbours(W, h=2, top_links=0.1)
        links = model.predict()
        top_links = W.count_nonzero() * 0.1
        for itr, vertices in enumerate(sorted(links, key=links.get, reverse=True)):
            i, j = vertices
            if itr > top_links:
                break
            W[i, j] = links[(i, j)]

        W = 0.5 * (W + W.T) #make symmetric

            # agora extrai edge_index (não direcionado)
        W_coo = W.tocoo()
        edge_index = torch.tensor([W_coo.row, W_coo.col], dtype=torch.long)

        return edge_index



args = utils.parse_arguments()
if args.config:
        config_params = utils.load_config_from_json(args.config)
        # Atualiza os parâmetros do argparse com os valores do JSON
        for key, value in config_params.items():
            setattr(args, key, value)

graph_generator_list = args.graph_generator.split(' ')

if args.run_both_datasets:
    if args.run_both_datasets == 'original':
        X = np.load(args.embedding_original_path)
        path = f'results/{args.actual_date}/original_graphs/'
    if args.run_both_datasets == 'debiased':
        X = np.load(args.embedding_debiased_path)
        path = f'results/{args.actual_date}/debiased_graphs/'

    else: 
        raise "UNDEFINED EMBED"
    for graph in graph_generator_list:
        #TODO: colocar o k nos parâmetros
        edge_index = graph_generator(X = X, graph_type = graph, k = 3)
        np.save(path + f'edge_index_{graph}', edge_index)

if args.run_both_datasets == None:
    raise "NOT IMPLEMENTED YET"

