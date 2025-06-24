
# Positive and Unlabeled Graph Learning with Large Language Models for Fake News Detection

Este repositório apresenta a implementação de uma pipeline do artigo *Positive and Unlabeled Graph Learning with Large Language Models for Fake News Detection* aceito no 35th BRACIS, voltada para a avaliação e mitigação de viés em modelos de linguagem de larga escala (LLMs) no contexto da detecção de notícias falsas. A proposta do projeto é investigar como técnicas de debiasing aplicadas a textos impactam a distribuição vetorial dos dados, a estrutura de grafos derivados dos embeddings e a performance em tarefas de agrupamento.

## Estrutura do Projeto

- `benchmark.py`: Executa os testes de performance comparativa entre diferentes embeddings e estratégias.
- `benchmark_analysis.py`: Analisa os resultados obtidos nos benchmarks e gera estatísticas descritivas.
- `data_processing.py`: Realiza o pré-processamento textual e limpeza dos dados.
- `dataset_analysis.py`: Fornece estatísticas sobre o conjunto de dados original e transformado.
- `text_embedding.py`: Responsável pela vetorização dos textos utilizando modelos LLM.
- `ollama_news_debias.py`: Aplica técnicas de debiasing textual utilizando LLMs locais via Ollama.
- `graph_generator.py`: Constrói grafos a partir dos embeddings utilizando k-NN.
- `graph_analysis.py`: Avalia propriedades estruturais dos grafos, como densidade, grau e componentes conexas.
- `run.sh`: Script auxiliar para execução da pipeline completa.
- `code_workflow.txt`, `workflow.svg`: Documentação e diagrama da sequência de execução dos módulos.
- `English_sample.txt`: Exemplo de entrada textual.
- `json_inputs/`: Arquivos de entrada no formato JSON utilizados na etapa de debiasing.
- `results/`: Diretório destinado aos resultados dos experimentos (atualmente vazio por limitação de espaço).

## Argumentos

| Argumento | Tipo | Descrição |
|-----------|------|-----------|
| `--config` | `str` | Path to the JSON config file with parameters |
| `--input_data_path` | `str` | Path to dataset in csv format |
| `--input_debiased_data_path` | `str` |  |
| `--llm_model` | `str` | OllaMa model name |
| `--prompt_debias_input_path` | `str` | path to .md file with prompt |
| `--prompt_summarization_input_path` | `str` | path to .md file with prompt |
| `--number_of_samples` | `int` | number of samples to run in test cases |
| `--actual_date` | `str` |  |
| `--embedding_model` | `str` |  |
| `--run_both_datasets` | `str` | variavel para execução do embedding dos textos. original indica que o conjunto de dados original será feito embedding, debiased indica que o conjunto de dados desinviesado será feito embedding, None aplica o embedding a ambos os datasets |
| `--embedding_original_path` | `str` |  |
| `--embedding_debiased_path` | `str` |  |
| `--graph_generator` | `` |  |
| `--benchmark_samples` | `int` |  |
| `--p` | `float` | portion of positive elements that will be used for train |
| `--models` | `str` | name of each models from model.model.py file. |
| `--input_column` | `str` | csv Column that the model will use to debias |
| `--input_instruction` | `str` | instruction of the model to debias |
| `--output_path` | `str` | output path and name |
| `--seed` | `int` | seed for sample purpouses |
| `--temperature` | `float` | LLM temperature |

## Requisitos

Recomenda-se a utilização de ambiente virtual Python. Para instalar as dependências:

```bash
pip install -r requirements.txt
```

Além disso, é necessário possuir o ambiente Ollama devidamente configurado para execução local de LLMs.

## Execução

As etapas principais podem ser realizadas conforme a sequência abaixo:

1. Geração dos embeddings:

```bash
python text_embedding.py
```

2. Aplicação do debiasing com modelo local:

```bash
python ollama_news_debias.py
```

3. Geração dos grafos:

```bash
python graph_generator.py
```

4. Avaliação dos resultados:

```bash
python benchmark.py
python benchmark_analysis.py
```

5. Análises complementares (opcional):

```bash
python dataset_analysis.py
python graph_analysis.py
```

## Objetivo da Pesquisa

O projeto visa compreender:

- Como o debiasing textual impacta os embeddings gerados por LLMs.
- De que maneira essas alterações afetam a estrutura dos grafos construídos a partir dos embeddings.
- Quais os efeitos sobre os agrupamentos formados, em especial no contexto de classificadores weakly supervised ou baseados em clusters.

## Tecnologias Utilizadas

- Python 3
- NumPy, Pandas, Scikit-learn
- NetworkX
- Matplotlib
- Ollama com LLM local (por exemplo, DeepSeek, Mistral)

## Observações

Os arquivos da pasta `results/` foram omitidos por questões de tamanho. Recomenda-se a execução completa da pipeline para geração dos resultados e análise das saídas.
