'''
Code for benchmark over the models/models.py file
'''

import utils.utils as utils
import torch
from models.models import *

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
    data = torch.load(f'results/{args.actual_date}/debiased_graphs/graph_{graph}.pt')

    for s in range(args.benchmark_samples):
        data.train_mask = torch.load(f'results/{args.actual_date}/debiased_graphs/samples/train_mask_{s}.pt')
        data.test_mask = torch.load(f'results/{args.actual_date}/debiased_graphs/samples/test_mask_{s}.pt')

        for model_name in model_list:
            if model_name == "MCLS":
                # TODO: hard coded, do it with args instead
                model = MCLS(data, k = 7, ratio = 0.3)
                model.train
                RN = model.negative_inference(num_neg = 10)
                print(RN)


