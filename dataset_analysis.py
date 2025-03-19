import pandas as pd
import utils.utils as utils

args = utils.parse_arguments()
if args.config:
        config_params = utils.load_config_from_json(args.config)
        # Atualiza os parâmetros do argparse com os valores do JSON
        for key, value in config_params.items():
            setattr(args, key, value)

original_df = pd.read_csv(args.input_data_path, sep = '\t')
debiased_df = pd.read_csv(args.input_debiased_data_path, sep = '\t')

# contando a average number of words

# Contar o número de palavras em cada texto
original_df["word_count"] = original_df["news"].apply(lambda x: len(str(x).split()))
debiased_df["word_count"] = debiased_df["Model Answer"].apply(lambda x: len(str(x).split()))


# Calcular a média do número de palavras
media_palavras_input = original_df["word_count"].mean()
media_palavras_output = debiased_df["word_count"].mean()

print(f"Média de palavras por texto no dataset original: {media_palavras_input:.2f}")
print(f"Média de palavras por texto no dataset processado: {media_palavras_output:.2f}")

#TODO: outras análises pertinentes

#TODO: salvar os outputs em um arquivo separado