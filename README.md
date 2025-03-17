# Removing bias via LLMs for fake news detection

Nesse repositório serão feitos os experimentos do artigo [].

para executar o código

`python3 ollama_news_debias.py --args´

## Tabela de argumentos

| Argumento               | Tipo  | Descrição |
|-------------------------|-------|-----------|
| `--config`             | str   | Path to the JSON config file with parameters |
| `--input_data_path`    | str   | Path to dataset in csv format |
| `--input_column`       | str   | CSV Column that the model will use to debias |
| `--input_instruction`  | str   | Instruction of the model to debias (default: texto para identificar notícias falsas) |
| `--output_path`        | str   | Output path and name (default: `output.csv`) |
| `--number_of_samples`  | int   | Number of samples to run in test cases |
| `--llm_model`         | str   | OllaMa model name (default: `gemma3:12b`) |
| `--seed`              | int   | Seed for sample purposes (default: `None`) |
| `--prompt_input_path` | str   | Path to .md file with prompt |

## Como usar

## Instalação

## Licença

