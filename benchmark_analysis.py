'''
Essa parte do código vai fazer com que a gente possa analisar de maneira bem ampla, e sem rodar o código novamente, os resultados obtidos pelos benchmarks
'''
# import utils.utils as utils
# import torch
# import pandas as pd
# from sklearn.metrics import f1_score

# args = utils.parse_arguments()

# if args.config:
#         config_params = utils.load_config_from_json(args.config)
#         # Atualiza os parâmetros do argparse com os valores do JSON
#         for key, value in config_params.items():
#             setattr(args, key, value)

# graph_generator_list = args.graph_generator.split(' ')
# model_list = args.models.split(' ')
# Y = torch.tensor(pd.read_csv(f'results/{args.actual_date}/llm_processed_data.tsv', sep = '\t')['label'])

# for graph in graph_generator_list:
#     for s in range(args.benchmark_samples):
#         test_mask = torch.load(f'results/{args.actual_date}/debiased_graphs/samples/test_mask_{s}.pt', weights_only=False)
#         y_true = torch.tensor(pd.read_csv(f'results/{args.actual_date}/llm_processed_data.tsv', sep = '\t')['label'])[test_mask]
#         for model in model_list:
#             y_pred = torch.load(f'results/{args.actual_date}/benchmark_outputs/output_{model}_{graph}_{s}.pt', weights_only=False)
#             print(model, f1_score(y_true, y_pred))

import utils.utils as utils
import torch
import pandas as pd
from sklearn.metrics import f1_score

args = utils.parse_arguments()

if args.config:
    config_params = utils.load_config_from_json(args.config)
    for key, value in config_params.items():
        setattr(args, key, value)

graph_generator_list = args.graph_generator.split(' ')
model_list = args.models.split(' ')
Y = torch.tensor(pd.read_csv(f'results/{args.actual_date}/llm_processed_data.tsv', sep='\t')['label'])

# Lista para armazenar os resultados
results = []

for graph in graph_generator_list:
    for s in range(args.benchmark_samples):
        test_mask = torch.load(
            f'results/{args.actual_date}/debiased_graphs/samples/test_mask_{s}.pt',
            weights_only=False
        )
        y_true = torch.tensor(
            pd.read_csv(f'results/{args.actual_date}/llm_processed_data.tsv', sep='\t')['label']
        )[test_mask]

        for model in model_list:
            y_pred = torch.load(
                f'results/{args.actual_date}/benchmark_outputs/output_{model}_{graph}_{s}.pt',
                weights_only=False
            )
            score = f1_score(y_true, y_pred)
            # Armazenando no formato de dicionário
            results.append({
                'model': model,
                'graph': graph,
                'sample': s,
                'f1_score': score
            })

# Convertendo para DataFrame
df_results = pd.DataFrame(results)

# Agrupa por modelo e grafo, calcula média e desvio padrão
grouped = df_results.groupby(['model', 'graph'])['f1_score'].agg(['mean', 'std']).reset_index()

# Renomeia colunas para clareza
grouped = grouped.rename(columns={
    'mean': 'f1_score_mean',
    'std': 'f1_score_std'
})

# Exporta para LaTeX
latex_table = grouped.to_latex(
    index=False,
    float_format="%.3f",
    caption="Resultados médios de F1-score com desvio padrão por modelo e grafo",
    label="tab:benchmark_results"
)
grouped.to_csv(f'results/{args.actual_date}/final_results.csv')
print(latex_table)


