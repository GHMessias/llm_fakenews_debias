import pandas as pd
import utils.utils as utils
import random
from collections import Counter
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import nltk
import networkx as nx
import numpy as np

args = utils.parse_arguments()
if args.config:
        config_params = utils.load_config_from_json(args.config)
        # Atualiza os parâmetros do argparse com os valores do JSON
        for key, value in config_params.items():
            setattr(args, key, value)

options = ['debiased', 'original']

def number_of_edges(edge_index):
      return edge_index.shape[1]

def density(edge_index, number_of_vertices):
      return edge_index.shape[1] / number_of_vertices ** 2

def contar_componentes_conexos(edge_index: np.ndarray) -> int:
    """
    Retorna o número de componentes conexos de um grafo definido pelo edge_index.

    Parâmetros:
    - edge_index: np.ndarray de shape (2, N), onde cada coluna representa uma aresta.

    Retorna:
    - int: número de componentes conexos (1 se o grafo for conexo).
    """

    if edge_index.shape[0] != 2:
        raise ValueError("edge_index deve ter shape (2, N)")

    G = nx.Graph()
    edges = edge_index.T.tolist()  # cada linha é uma aresta (u, v)
    G.add_edges_from(edges)

    return nx.number_connected_components(G)

graph_list = args.graph_generator.split(' ')
for opt in options:
    # ler o edge_index dos grafos
    # 1. fazer uma função que realiza a contagem da quantidade de vértices de cada edge_index
    # 2. fazer uma função que verifica se aqueles edge_index são conexos, ou quantos componentes desconexos tem
    # 3. fazer uma função que calcula a densidade dado o número de arestas e o número de vértices (a densidade é a quantidade de arestas dividida pelo quadrado dos vértices)
    for graph in graph_list:
         edge_index = np.load(f'results/{args.actual_date}/{opt}_graphs/edge_index_{graph}.npy')
         print(f'number of edges of graph {graph} {opt}: {number_of_edges(edge_index)}')
         print(f'density of graph {graph} {opt}: {density(edge_index, 2064)}')
         print(f'number of connected components of graph {graph} {opt}: {contar_componentes_conexos(edge_index)}')