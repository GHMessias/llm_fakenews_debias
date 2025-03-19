import pandas as pd
import ollama
import utils.utils as utils
import numpy as np


def get_embedding(text):
    response = ollama.embeddings(model="nomic-embed-text", prompt=text) 
    return response['embedding']

args = utils.parse_arguments()
if args.config:
        config_params = utils.load_config_from_json(args.config)
        # Atualiza os parâmetros do argparse com os valores do JSON
        for key, value in config_params.items():
            setattr(args, key, value)

print(args.embed_both_datasets)

if args.embed_both_datasets:
    if args.embed_both_datasets == 'original':
        df = pd.read_csv(args.input_data_path, sep = '\t')
    if args.embed_both_datasets == 'debiased':
        df = pd.read_csv(args.input_debiased_data_path, sep = '\t')
    else: 
        raise "UNDEFINED EMBED"
    df['embedding'] = df['news'].apply(get_embedding)
    embeddings_array = np.array(df['embedding'].tolist())
    output_path = f'results/{args.actual_date}/embedded_{args.embed_both_datasets}_data.npy'
    np.save(output_path, embeddings_array)

if args.embed_both_datasets == None:
    df_original = pd.read_csv(args.input_data_path, sep = '\t')
    df_debiased = pd.read_csv(args.input_debiased_data_path, sep = '\t')


    df_original['embedding'] = df_original['news'].apply(get_embedding)
    df_debiased['embedding'] = df_debiased['news'].apply(get_embedding)

    # Converter os embeddings para um numpy array
    embeddings_array_original = np.array(df_original['embedding'].tolist())
    embeddings_array_debiased = np.array(df_debiased['embedding'].tolist())

    output_path_original = f'results/{args.actual_date}/embedded_original_data.npy'
    output_path_debiased = f'results/{args.actual_date}/embedded_debiased_data.npy'
    # Salvar o numpy array como arquivo .npy
    np.save(output_path_original, embeddings_array_original)
    np.save(output_path_debiased, embeddings_array_debiased)

