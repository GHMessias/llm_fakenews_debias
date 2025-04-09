'''
Code for benchmark over the models/models.py file
'''

import utils.utils as utils
from utils.rewiring import rewiring
from utils.gae_functions import train_gae, gae_negative_inference
import torch
from models.models import *
from torch_geometric.nn import GAE
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score


args = utils.parse_arguments()
if args.config:
        config_params = utils.load_config_from_json(args.config)
        # Atualiza os parâmetros do argparse com os valores do JSON
        for key, value in config_params.items():
            setattr(args, key, value)


graph_generator_list = args.graph_generator.split(' ')
model_list = args.models.split(' ')

# TODO: alterar para lidar com ambos os tipos de dados, original e debiased
for graph in graph_generator_list:
    data = torch.load(f'results/{args.actual_date}/debiased_graphs/graph_{graph}.pt', weights_only=False)
    data.y = torch.tensor(pd.read_csv(f'results/{args.actual_date}/llm_processed_data.tsv', sep = '\t')['label'])

    for s in range(args.benchmark_samples):
        # TODO: remover o hard coded, usado somente para testes pontuais

        # data.train_mask = torch.load(f'results/2025-04-01_17:56:43/debiased_graphs/samples/train_mask_{s}.pt')
        # data.test_mask = torch.load(f'results/2025-04-01_17:56:43/debiased_graphs/samples/test_mask_{s}.pt')
        data.train_mask = torch.load(f'results/{args.actual_date}/debiased_graphs/samples/train_mask_{s}.pt')
        data.test_mask = torch.load(f'results/{args.actual_date}/debiased_graphs/samples/test_mask_{s}.pt')

        # montar os conjuntos P e U
        data.P = torch.nonzero(data.train_mask, as_tuple=True)[0]
        data.U = torch.nonzero(data.test_mask, as_tuple=True)[0]

        for model_name in model_list:
            # Pensando em remover pois está dando resultados vazios
            if model_name == "MCLS":
                # TODO: hard coded, do it with args instead
                model = MCLS(data, k = 2, ratio = 0.3)
                model.train()
                RN = model.negative_inference(num_neg = 10)

                indices = data.P.tolist() + RN
                
                y_train_SVM = np.array([1] * len(data.P) + [-1] * len(RN))

                svm_clf = SVC(kernel = 'rbf', C = 1.0, gamma = 'scale')
                svm_clf.fit(data.x[indices], y_train_SVM)
                y_pred = torch.tensor(svm_clf.predict(data.x[data.test_mask]))
                torch.save(y_pred, f'results/{args.actual_date}/benchmark_outputs/output_{model_name}_{s}.pt')
                

            if model_name == 'CCRNE':
                # TODO: hard coded, do it with args instead
                model = CCRNE(data, ratio = 0.3)
                model.train()
                RN = model.negative_inference(num_neg = 3)

                indices = data.P.tolist() + RN.tolist()
                
                y_train_SVM = np.array([1] * len(data.P) + [-1] * len(RN))

                svm_clf = SVC(kernel = 'rbf', C = 1.0, gamma = 'scale')
                svm_clf.fit(data.x[indices], y_train_SVM)
                y_pred = torch.tensor(svm_clf.predict(data.x[data.test_mask]))
                torch.save(y_pred, f'results/{args.actual_date}/benchmark_outputs/output_{model_name}_{s}.pt')

      
            if model_name == 'PU_LP':
                # TODO: hard coded, do it with args instead
                model = PU_LP(data = data, alpha = 0.1, m = 3, l = 1)
                model.train()
                RN = model.negative_inference(num_neg = 3)
            
                indices = data.P.tolist() + RN.tolist()
                
                y_train_SVM = np.array([1] * len(data.P) + [-1] * len(RN))

                svm_clf = SVC(kernel = 'rbf', C = 1.0, gamma = 'scale')
                svm_clf.fit(data.x[indices], y_train_SVM)
                y_pred = torch.tensor(svm_clf.predict(data.x[data.test_mask]))
                torch.save(y_pred, f'results/{args.actual_date}/benchmark_outputs/output_{model_name}_{s}.pt')


            if model_name == 'LP_PUL':
                model = LP_PUL(data)
                model.train()
                RN = model.negative_inference(num_neg = 10)

                indices = data.P.tolist() + RN.tolist()
                
                y_train_SVM = np.array([1] * len(data.P) + [-1] * len(RN))

                svm_clf = SVC(kernel = 'rbf', C = 1.0, gamma = 'scale')
                svm_clf.fit(data.x[indices], y_train_SVM)
                y_pred = torch.tensor(svm_clf.predict(data.x[data.test_mask]))
                torch.save(y_pred, f'results/{args.actual_date}/benchmark_outputs/output_{model_name}_{s}.pt')


            if model_name == 'RCSVM':
                model = RCSVM(data = data, alpha = 0.7, beta = 0.3)
                model.train()
                RN = model.negative_inference(num_neg = 10)

                indices = data.P.tolist() + RN.tolist()
                
                y_train_SVM = np.array([1] * len(data.P) + [-1] * len(RN))

                svm_clf = SVC(kernel = 'rbf', C = 1.0, gamma = 'scale')
                svm_clf.fit(data.x[indices], y_train_SVM)
                y_pred = torch.tensor(svm_clf.predict(data.x[data.test_mask]))
                torch.save(y_pred, f'results/{args.actual_date}/benchmark_outputs/output_{model_name}_{s}.pt')

            # TODO: Modelo GCN e RewiringGCN

            if model_name == 'PSRB':
                # Precisa aplicar a reescrita nos dados
                # TODO: Hard coded, do it with args instead
                model = GAE(encoder = RGCN(input_size = data.x.shape[1], hidden_size=64, output_size=32, L = 2))
                data.graph_list = rewiring(graph = to_networkit(data.edge_index, directed=False), L = 3, P = data.P)

                # Precisa chamar o treinamento do modelo
                optimizer = torch.optim.Adam(params=model.parameters(), lr = 0.001) 
                train_gae(data = data, gae_model = model, optimizer = optimizer, epochs = 100, verbose = True)
                RN = gae_negative_inference(data, model, len(data.P))
                print(RN)
                # Precisa fazer a fase de inferência

                indices = data.P.tolist() + RN.tolist()
                
                y_train_SVM = np.array([1] * len(data.P) + [-1] * len(RN))

                svm_clf = SVC(kernel = 'rbf', C = 1.0, gamma = 'scale')
                svm_clf.fit(data.x[indices], y_train_SVM)
                y_pred = torch.tensor(svm_clf.predict(data.x[data.test_mask]))
                torch.save(y_pred, f'results/{args.actual_date}/benchmark_outputs/output_{model_name}_{s}.pt')
            
