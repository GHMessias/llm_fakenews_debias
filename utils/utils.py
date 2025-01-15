import argparse
import json

def parse_arguments():
    '''
    Function to collect the arguments
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to the JSON config file with parameters')
    parser.add_argument('--input_data_path', type = str, help = 'Path to dataset in csv format')
    parser.add_argument('--input_column', type = str, help = 'csv Column that the model will use to debias')
    parser.add_argument('--input_instruction', type = str, help = 'instruction of the model to debias', default = 'Suponha que você está trabalhando para identificar notícias falsas a partir de um site de checagem. Seu objetivo é, dado texto do site de checagem, gerar a possível notícia falsa que foi divulgada. Seja direto, respondendo somente a notícia falsa em questão, sem justificativas ou textos excedentes. Sabendo disso, identifique a notícia falsa que foi divulgada a partir do seguinte texto do site de checagem: ')
    parser.add_argument('--output_path', type = str, help = 'output path and name', default = 'output.csv')
    parser.add_argument('--number_of_samples', type = int, help = 'number of samples to run in test cases')
    parser.add_argument('--llm_model', type = str, default = 'llama3.1', help = 'OllaMa model name')
    parser.add_argument('--seed', type = int, default = 42, help = 'seed for sample purpouses')

    return parser.parse_args()

def load_config_from_json(json_file):
    '''Function to load parameters from a JSON file'''
    with open(json_file, 'r') as f:
        config = json.load(f)
    return config