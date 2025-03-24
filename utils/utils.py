import argparse
import json

def parse_arguments():
    '''
    Function to collect the arguments
    '''

    # TODO: Colocar um parâmetro para truncar o dataset de input
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to the JSON config file with parameters')
    parser.add_argument('--input_data_path', type = str, help = 'Path to dataset in csv format')
    parser.add_argument('--input_debiased_data_path', type = str)
    parser.add_argument('--llm_model', type = str, default = 'gemma3:12b', help = 'OllaMa model name')
    parser.add_argument('--prompt_debias_input_path', type = str, help = "path to .md file with prompt")
    parser.add_argument('--prompt_summarization_input_path', type = str, help = "path to .md file with prompt")
    parser.add_argument('--number_of_samples', type = int, help = 'number of samples to run in test cases')
    parser.add_argument('--actual_date', type = str)
    parser.add_argument('--embedding_model', type = str)
    parser.add_argument('--run_both_datasets', type = str, default = None, help = "variavel para execução do embedding dos textos. original indica que o conjunto de dados original será feito embedding, debiased indica que o conjunto de dados desinviesado será feito embedding, None aplica o embedding a ambos os datasets")
    parser.add_argument('--embedding_original_path', type = str)
    parser.add_argument('--embedding_debiased_path', type = str)
    parser.add_argument('--graph_generator', nargs = '+')
    parser.add_argument('--benchmark_samples', type = int, default = 2)


    parser.add_argument('--input_column', type = str, help = 'csv Column that the model will use to debias')
    parser.add_argument('--input_instruction', type = str, help = 'instruction of the model to debias', default = 'Suponha que você está trabalhando para identificar notícias falsas a partir de um site de checagem. Seu objetivo é, dado texto do site de checagem, gerar a possível notícia falsa que foi divulgada. Seja direto, respondendo somente a notícia falsa em questão, sem justificativas ou textos excedentes. Sabendo disso, identifique a notícia falsa que foi divulgada a partir do seguinte texto do site de checagem: ')
    parser.add_argument('--output_path', type = str, help = 'output path and name', default = 'output.csv')
    parser.add_argument('--seed', type = int, default = None, help = 'seed for sample purpouses')
    parser.add_argument('--temperature', type = float, default = 0.1, help = "LLM temperature")

    return parser.parse_args()

def load_config_from_json(json_file):
    '''Function to load parameters from a JSON file'''
    with open(json_file, 'r') as f:
        config = json.load(f)
    return config