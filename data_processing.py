'''
This file is responsable for create the torch_geeometric.data file, split into test (for positive samples only) and train. The output of this file is: the torch_gemetric.data file with X, edge_index and Y. The `samples` files with each sample.
'''

import numpy as np
import torch
from torch_geometric.data import Data
import utils.utils as utils
import pandas as pd

args = utils.parse_arguments()
if args.config:
        config_params = utils.load_config_from_json(args.config)
        # Atualiza os parâmetros do argparse com os valores do JSON
        for key, value in config_params.items():
            setattr(args, key, value)

# label value
y = pd.read_csv(args.input_data_path, sep = '\t')['label'].to_numpy()

# gerar os 
if args.run_both_datasets == None or args.run_both_datasets == 'original':
    # para cada um dos grafos gerados vamos rodar um conjunto dos dados
    X = np.load(args.embedding_original_path)

    

if args.run_both_datasets == None or args.run_both_datasets == 'debiased':
    X = np.load(args.embedding_debiased_path)

    # Gerando o edge_index de cada grafo a partir dos arquivos criados em graph_generator.py
    graph_generator_list = args.graph_generator.split(' ')

    for graph in graph_generator_list:
        edge_index = np.load(f'results/{args.actual_date}/debiased_graphs/edge_index_{graph}.npy')
        # Converte para tensores do PyTorch
        x_tensor = torch.tensor(X, dtype=torch.float)  # Nós (features)
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)  # Arestas
        y_tensor = torch.tensor(y, dtype = torch.long)
        # Filtra os índices dos exemplos positivos (classe 1)
        positive_indices = np.where(y == 1)[0]
        graph_data = Data(x=x_tensor, edge_index=edge_index_tensor, y=y_tensor)

        torch.save(graph_data, f'results/{args.actual_date}/debiased_graphs/graph_{graph}.pt')

        for s in range(args.benchmark_samples):
             # Filtra os índices dos exemplos positivos (classe 1)
            positive_indices = np.where(y == 1)[0]

            # Aleatoriza os índices positivos
            np.random.shuffle(positive_indices)
            split_idx = int(len(positive_indices) * args.p)
            train_indices = positive_indices[:split_idx]  # p% para treino
            test_indices = np.setdiff1d(np.arange(len(y)), train_indices)  # O restante para teste

            # Cria máscaras booleanas para treino e teste
            train_mask = torch.zeros(len(y), dtype=torch.bool)
            test_mask = torch.ones(len(y), dtype=torch.bool)

            # Define quais nós fazem parte do treino
            train_mask[train_indices] = True
            # Define quais nós fazem parte do teste
            test_mask[test_indices] = True

            # TODO: organizar as pastas onde os arquivos serão salvos, também seus caminhos e diferenças
            torch.save(train_mask, f'results/{args.actual_date}/debiased_graphs/samples/train_mask_{s}.pt')
            torch.save(test_mask, f'results/{args.actual_date}/debiased_graphs/samples/test_mask_{s}.pt')





# edge_index = np.load("edge_index_KNN.npy")  # Arestas do grafo
# # TODO: colocar o arquivo de rótulos Y
# # y = np.load('y.npy')
# y = np.array([1] * 10)

# # Converte para tensores do PyTorch
# x_tensor = torch.tensor(X, dtype=torch.float)  # Nós (features)
# edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)  # Arestas
# y_tensor = torch.tensor(y, dtype = torch.long)

# # Filtra os índices dos exemplos positivos (classe 1)
# positive_indices = np.where(y == 1)[0]

# graph_data = Data(x=x_tensor, edge_index=edge_index_tensor, y=y_tensor)

# # TODO: organizar o loop for em variavel args
# samples = 2

# # Esse loop for é responsável por gerar `samples` diferentes que serão usados para teste.
# for s in range(args.samples):
#     # TODO: colocar o valor de p em variavel args
#     p = 0.3

#     # Filtra os índices dos exemplos positivos (classe 1)
#     positive_indices = np.where(y == 1)[0]

#     # Aleatoriza os índices positivos
#     np.random.shuffle(positive_indices)
#     split_idx = int(len(positive_indices) * p)
#     train_indices = positive_indices[:split_idx]  # p% para treino
#     test_indices = np.setdiff1d(np.arange(len(y)), train_indices)  # O restante para teste

#     # Cria máscaras booleanas para treino e teste
#     train_mask = torch.zeros(len(y), dtype=torch.bool)
#     test_mask = torch.ones(len(y), dtype=torch.bool)

#     # Define quais nós fazem parte do treino
#     train_mask[train_indices] = True
#     # Define quais nós fazem parte do teste
#     test_mask[test_indices] = True

#     # TODO: organizar as pastas onde os arquivos serão salvos, também seus caminhos e diferenças
#     torch.save(train_mask, f'train_mask_{s}.pt')
#     torch.save(test_mask, f'test_mask_{s}.pt')
    