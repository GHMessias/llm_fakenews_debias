import pandas as pd
import utils.utils as utils
import random
from collections import Counter
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords


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

debiased_df_fake = debiased_df[debiased_df['label'] == 1]

# Carregar as stopwords do nltk
stop_words = set(stopwords.words('portuguese'))

# Concatenar todas as notícias para criar uma string
all_words = ' '.join(debiased_df_fake['Model Answer'].dropna())
words = re.findall(r'\w+', all_words.lower())

# Remover stopwords da lista de palavras
filtered_words = [word for word in words if word not in stop_words]

# Contar as palavras após a remoção das stopwords
word_counts = Counter(filtered_words)

# Filtrar palavras que aparecem mais de 5 vezes
filtered_word_counts = {word: count for word, count in word_counts.items() if count > 3}

# Gerar a wordcloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(filtered_words))

# Plotar a wordcloud
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.tight_layout()
plt.savefig(f'results/{args.actual_date}/dataset_analysis/wordcloud.png')  # Salva o gráfico
plt.close()

# Ordenar por frequência decrescente
sorted_words = dict(sorted(filtered_word_counts.items(), key=lambda item: item[1], reverse=True))

# Criar DataFrame para plotar
freq_df = pd.DataFrame(list(sorted_words.items()), columns=['Palavra', 'Frequência'])

# Plotar gráfico de barras horizontal
plt.figure(figsize=(12, 8))
plt.barh(freq_df['Palavra'], freq_df['Frequência'])
plt.gca().invert_yaxis()  # palavras mais frequentes no topo
plt.xlabel('Frequência')
plt.title('Palavras que aparecem mais de 5 vezes no dataset debiased (fake_news)')
plt.tight_layout()
plt.show()
plt.savefig(f'results/{args.actual_date}/dataset_analysis/word_distribution_false_debiased.png')  # Salva no diretório atual como .png
plt.close()

######################################################################################################################################

original_df_fake = debiased_df[debiased_df['label'] == 1]

# Concatenar todas as notícias para criar uma string
all_words = ' '.join(original_df_fake['news'].dropna())
words = re.findall(r'\w+', all_words.lower())

# Remover stopwords da lista de palavras
filtered_words = [word for word in words if word not in stop_words]

# Contar as palavras após a remoção das stopwords
word_counts = Counter(filtered_words)

# Filtrar palavras que aparecem mais de 5 vezes
filtered_word_counts = {word: count for word, count in word_counts.items() if count >= 75}

# Ordenar por frequência decrescente
sorted_words = dict(sorted(filtered_word_counts.items(), key=lambda item: item[1], reverse=True))

# Criar DataFrame para plotar
freq_df = pd.DataFrame(list(sorted_words.items()), columns=['Palavra', 'Frequência'])

# Plotar gráfico de barras horizontal
plt.figure(figsize=(12, 8))
plt.barh(freq_df['Palavra'], freq_df['Frequência'])
plt.gca().invert_yaxis()  # palavras mais frequentes no topo
plt.xlabel('Frequência')
plt.title('Palavras que aparecem mais de tantas vezes no dataset original (fake_news)')
plt.tight_layout()
plt.show()
plt.savefig(f'results/{args.actual_date}/dataset_analysis/word_distribution_false_original.png')  # Salva no diretório atual como .png
plt.close()

# TODO: 2. comparar a distribuição verdadeiras depois e falsas depois

debiased_df_true = debiased_df[debiased_df['label'] == -1]

all_words = ' '.join(debiased_df_true['Model Answer'].dropna())
words = re.findall(r'\w+', all_words.lower())

# Remover stopwords da lista de palavras
filtered_words = [word for word in words if word not in stop_words]

# Contar as palavras após a remoção das stopwords
word_counts = Counter(filtered_words)

# Filtrar palavras que aparecem mais de 5 vezes
filtered_word_counts = {word: count for word, count in word_counts.items() if count > 10}

# Ordenar por frequência decrescente
sorted_words = dict(sorted(filtered_word_counts.items(), key=lambda item: item[1], reverse=True))

# Criar DataFrame para plotar
freq_df = pd.DataFrame(list(sorted_words.items()), columns=['Palavra', 'Frequência'])

# Plotar gráfico de barras horizontal
plt.figure(figsize=(12, 8))
plt.barh(freq_df['Palavra'], freq_df['Frequência'])
plt.gca().invert_yaxis()  # palavras mais frequentes no topo
plt.xlabel('Frequência')
plt.title('Palavras que aparecem mais de 5 vezes no dataset debiased (true_news)')
plt.tight_layout()
plt.show()
plt.savefig(f'results/{args.actual_date}/dataset_analysis/word_distribution_true_debiased.png')  # Salva no diretório atual como .png
plt.close()
