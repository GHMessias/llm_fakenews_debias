import pandas as pd
import utils.utils as utils
import random
from collections import Counter
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import nltk


args = utils.parse_arguments()
if args.config:
        config_params = utils.load_config_from_json(args.config)
        # Atualiza os parâmetros do argparse com os valores do JSON
        for key, value in config_params.items():
            setattr(args, key, value)

original_df = pd.read_csv(args.input_data_path, sep = '\t')
debiased_df = pd.read_csv(args.input_debiased_data_path, sep = '\t')

# Adiciona contagem de palavras
original_df["word_count"] = original_df["news"].apply(lambda x: len(str(x).split()))
debiased_df["word_count"] = debiased_df["Model Answer"].apply(lambda x: len(str(x).split()))

# Calcula a média de palavras para cada label no dataset original
media_original_por_label = original_df.groupby("label")["word_count"].mean()

# Calcula a média de palavras para cada label no dataset debiased
media_debiased_por_label = debiased_df.groupby("label")["word_count"].mean()

# Exibe os resultados
print("Média de palavras por rótulo no dataset original:")
for label, media in media_original_por_label.items():
    print(f"Label {label}: {media:.2f} palavras")

print("\nMédia de palavras por rótulo no dataset processado (debiased):")
for label, media in media_debiased_por_label.items():
    print(f"Label {label}: {media:.2f} palavras")

# Comparar os antes e depois (verdadeiro_antes, falso_antes com verdadeiro_depois, falso_depois)

# Comparar entre verdadeiros depois e falsos depois
# 1. Considerando que a tarefa principal é remover o viés com LLMs, elas foram capazes de desinviesar o conjunto de dados? 2. Se sim, o novo conjunto de dados é comparável com notícias verdadeiras e falsas?

# TODO: 1. Colocar 3 samples do dataset antes e depois para fakenews

# Exibir amostras

random_state = random.randint(0, 10000)

original_sample = original_df[['news', 'label']].sample(n=3, random_state=random_state)
debiased_sample = debiased_df[['Model Answer', 'label']].sample(n=3, random_state=random_state)

# Salvar as amostras em um DataFrame
sample_df = pd.DataFrame({
    'Original Text': original_sample['news'],
    'Original Label': original_sample['label'],
    'Debiased Text': debiased_sample['Model Answer'],
    'Debiased Label': debiased_sample['label']
})

sample_df.to_csv(f'results/{args.actual_date}/dataset_analysis/samples_input_output.csv', sep = '\t')

# TODO: 1. Distribuição das palavras / palavras que mais apareceram / wordcloud / como estão as palavras que foram removidas por caravant et. al 2021 (em textos falsos)

# Garante que as stopwords estejam disponíveis
nltk.download("stopwords")

def gerar_wordcloud(caminho_salvar: str, dataframe: pd.DataFrame, nome_coluna: str):
    """
    Gera e salva uma wordcloud a partir dos textos de uma coluna de um DataFrame,
    removendo stopwords em português.

    Parâmetros:
    - caminho_salvar: str. Caminho para salvar a imagem gerada (ex: 'saida/wordcloud.png').
    - dataframe: pd.DataFrame. DataFrame contendo a coluna de textos.
    - nome_coluna: str. Nome da coluna que contém os textos.
    """

    if nome_coluna not in dataframe.columns:
        raise ValueError(f"A coluna '{nome_coluna}' não foi encontrada no DataFrame.")

    # Carregar stopwords em português
    stop_words = set(stopwords.words("portuguese"))

    # Concatenar todos os textos da coluna
    texto_total = " ".join(dataframe[nome_coluna].dropna().astype(str).tolist())

    # Limpar e remover stopwords
    palavras = re.findall(r'\w+', texto_total.lower())  # remover pontuação e baixar caixa
    palavras_filtradas = [palavra for palavra in palavras if palavra not in stop_words]
    texto_filtrado = " ".join(palavras_filtradas)

    # Gerar a wordcloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        max_words=200,
        collocations=False
    ).generate(texto_filtrado)

    # Plotar e salvar
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(caminho_salvar)
    plt.close()

    print(f"Wordcloud salva em: {caminho_salvar}")

def plotar_palavras_frequentes(dataframe: pd.DataFrame, nome_coluna: str, n: int = 20, caminho_salvar: str = None):
    """
    Plota (e opcionalmente salva) um gráfico com as n palavras mais frequentes da coluna de texto,
    removendo stopwords em português.

    Parâmetros:
    - dataframe: pd.DataFrame. DataFrame contendo a coluna de textos.
    - nome_coluna: str. Nome da coluna que contém os textos.
    - n: int. Número de palavras mais frequentes a serem exibidas.
    - caminho_salvar: str, opcional. Caminho para salvar o gráfico (ex: 'saida/top_palavras.png').
                      Se None, o gráfico será exibido na tela.
    """

    if nome_coluna not in dataframe.columns:
        raise ValueError(f"A coluna '{nome_coluna}' não foi encontrada no DataFrame.")

    # Stopwords em português
    stop_words = set(stopwords.words("portuguese"))
    stop_words.add('r')

    # Juntar os textos e remover stopwords
    texto_total = " ".join(dataframe[nome_coluna].dropna().astype(str).tolist())
    palavras = re.findall(r'\w+', texto_total.lower())
    palavras_filtradas = [palavra for palavra in palavras if palavra not in stop_words]

    # Contagem de palavras
    contagem = Counter(palavras_filtradas)
    palavras_comuns = contagem.most_common(n)

    if not palavras_comuns:
        print("Nenhuma palavra frequente encontrada.")
        return

    # Separar para plotagem
    palavras, frequencias = zip(*palavras_comuns)

    # Plotar gráfico
    plt.figure(figsize=(10, 6))
    plt.barh(palavras[::-1], frequencias[::-1])  # inverter para colocar a mais frequente no topo
    plt.xlabel("Frequência")
    plt.ylabel("Palavras")
    plt.title(f"Top {n} Palavras Mais Frequentes")
    plt.tight_layout()

    if caminho_salvar:
        plt.savefig(caminho_salvar)
        plt.close()
        print(f"Gráfico salvo em: {caminho_salvar}")
    else:
        plt.show()


gerar_wordcloud(caminho_salvar = f'results/{args.actual_date}/dataset_analysis/wordcloud_fake_original.png', dataframe = original_df[original_df['label'] == 1], nome_coluna='news')
gerar_wordcloud(caminho_salvar = f'results/{args.actual_date}/dataset_analysis/wordcloud_real_original.png', dataframe = original_df[original_df['label'] == -1], nome_coluna='news')
gerar_wordcloud(caminho_salvar = f'results/{args.actual_date}/dataset_analysis/wordcloud_real_debiased.png', dataframe = debiased_df[debiased_df['label'] == -1], nome_coluna='Model Answer')
gerar_wordcloud(caminho_salvar = f'results/{args.actual_date}/dataset_analysis/wordcloud_fake_debiased.png', dataframe = debiased_df[debiased_df['label'] == 1], nome_coluna='Model Answer')

plotar_palavras_frequentes(dataframe = original_df[original_df['label'] == 1], nome_coluna = 'news', n = 20, caminho_salvar = f'results/{args.actual_date}/dataset_analysis/wordcount_fake_original.png')
plotar_palavras_frequentes(dataframe = original_df[original_df['label'] == -1], nome_coluna = 'news', n = 20, caminho_salvar = f'results/{args.actual_date}/dataset_analysis/wordcount_real_original.png')
plotar_palavras_frequentes(dataframe = debiased_df[debiased_df['label'] == -1], nome_coluna = 'Model Answer', n = 20, caminho_salvar = f'results/{args.actual_date}/dataset_analysis/wordcount_real_debiased.png')
plotar_palavras_frequentes(dataframe = debiased_df[debiased_df['label'] == 1], nome_coluna = 'Model Answer', n = 20, caminho_salvar = f'results/{args.actual_date}/dataset_analysis/wordcount_fake_debiased.png')


# Fazer uma função que:
# 1. Gera uma wordcloud a partir do dataframe e da coluna selecionada (a coluna contém textos), salva no caminho especificado
# 2. Pega o dataframe e a coluna selecionada e encontra as n palavras que mais apareceram, juntamente com sua contagem. Faz um gráfico onde as palavras aparecem no eixo y e a contagem no eixo x