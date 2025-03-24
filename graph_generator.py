import utils.utils as utils
from sklearn.neighbors import NearestNeighbors
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree

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
         print('NOT IMPLEMENTED')
         return


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
    if args.run_both_datasets == 'debiased':
        X = np.load(args.embedding_debiased_path)
    else: 
        raise "UNDEFINED EMBED"
    for graph in graph_generator_list:
        #TODO: colocar o k nos parâmetros
        edge_index = graph_generator(X = X, graph_type = graph, k = 3)
        np.save(f'results/{args.actual_date}/edge_index_{graph}.npy', edge_index)

if args.run_both_datasets == None:
    raise "NOT IMPLEMENTED YET"

