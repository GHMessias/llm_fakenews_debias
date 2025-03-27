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

## Code Workflow

O código dos experimentos está organizado da seguinte forma:

Input: Conjunto de dados com notícias reais e notícias de blog de checagem. As notícias reais tem rótulo -1 enquanto as notícias falsas tem rótulo 1.

Objetivo: Classificar as notícias de maneira transdutiva semissupervisionada por meio do paradigma de aprendizado positivo. No caso desse trabalho, as notícias falsas (rótulo 1) serão utilizadas como classe positva durante o treinamento.

Arquivos:

1. llm_news_debias.py -- Esse arquivo é responsável por extrair as notícias falsas dos sites de checagem caso a notícia seja falsa ou executar uma sumarização caso a notícia seja verdadeira. O objetivo é deixar as notícias falsas e verdadeiras com um tamanho próximo.

2. data_analysis.py -- Arquivo que faz as análises do novo dataset (contagem de palavras, tokes, etc.)

3. text_embedding.py -- Faz a transformação dos textos em vetores

4. graph_generator.py -- Transforma o output do text_embedding.py em um grafo a partir de um algoritmo pré-definido

5. data_processing.py -- transforma o output do text_embeddings.py em um formato onde seja possível importar usando torch_geometric.data.

6. benchmark.py -- Aplica os modelos definidos em models.py no benchmark de dados positivos. Esse arquivo deve aplicar o grafo gerado no benchmark.



Textos feitos truncamento em 2000 caracteres