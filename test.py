import utils.utils as utils
import torch
import pandas as pd
from sklearn.metrics import f1_score
from utils.rewiring import rewiring
from torch_geometric.nn import GAE
from models.models import *
from utils.gae_functions import train_gae, gae_negative_inference
from sklearn.svm import SVC
from sklearn.metrics import f1_score

args = utils.parse_arguments()
if args.config:
    config_params = utils.load_config_from_json(args.config)
    for key, value in config_params.items():
        setattr(args, key, value)

# opt = 'debiased'

# test_mask= torch.load(f'results/2025-04-23_15:00:05/{opt}_graphs/samples/test_mask_0.pt', weights_only=False)
# y_pred = torch.load(f'results/2025-04-23_15:00:05/benchmark_outputs/{opt}/output_PSRB_yake_0.pt', weights_only=False)
# print(len(y_pred))
# y_true = torch.tensor(pd.read_csv(args.input_data_path, sep='\t')['label'])[test_mask]
# print(len(y_true))
# print(torch.unique(y_pred))
# print(torch.unique(y_true))
# print(f1_score(y_true, y_pred))
# print(y_true[:100])
# print(y_pred[:100])

# graph_generator_list = args.graph_generator.split(' ')
graph_generator_list = ['KNN']
model = 'PSRB'
data_type = 'debiased'
results = []

for graph in graph_generator_list:
    data = torch.load(f'results/{args.actual_date}/{data_type}_graphs/graph_{graph}.pt', weights_only=False)
    data.y = torch.tensor(pd.read_csv(args.input_debiased_data_path, sep = '\t')['label'])

    for s in range(args.benchmark_samples):
        data.train_mask = torch.load(f'results/{args.actual_date}/{data_type}_graphs/samples/train_mask_{s}.pt', weights_only=False)
        data.test_mask = torch.load(f'results/{args.actual_date}/{data_type}_graphs/samples/test_mask_{s}.pt', weights_only=False)

        data.P = torch.nonzero(data.train_mask, as_tuple=True)[0]
        num_neg = len(data.P)
        data.U = torch.nonzero(data.test_mask, as_tuple=True)[0]

        model = GAE(encoder = RGCN(input_size = data.x.shape[1], hidden_size=64, output_size=32, L = 2))
        data.graph_list = rewiring(graph = to_networkit(data.edge_index, directed=False), L = 3, P = data.P)

        optimizer = torch.optim.Adam(params=model.parameters(), lr = 0.001) 
        train_gae(data = data, gae_model = model, optimizer = optimizer, epochs = 100, verbose = True)
        RN = gae_negative_inference(data, model, len(data.P))
        # print(len([x for x in RN if x > 1032]) / len(RN))

        indices = data.P.tolist() + RN.tolist()
        y_train_SVM = np.array([1] * len(data.P) + [-1] * len(RN))

        svm_clf = SVC(kernel = 'rbf', C = 1.0, gamma = 'scale')
        svm_clf.fit(data.x[indices], y_train_SVM)
        y_pred = torch.tensor(svm_clf.predict(data.x[data.test_mask]))

        print(torch.unique(y_pred, return_counts = True))
        print('f1 score ', f1_score(data.y[data.test_mask], y_pred))
        results.append(f1_score(data.y[data.test_mask], y_pred))

        # print(y_pred[:50])
        print(data.y[data.test_mask][:50])

print(np.mean(results))
        