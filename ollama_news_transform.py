import pandas as pd
from ollama import chat
from ollama import ChatResponse

df = pd.read_csv('datasets/FactCheckedNews/Dataset/Fact_checked_news.tsv', sep = '\t', header = None)
df.columns = ['path', 'news', 'label']
df = df.head()

def query_olmo2(news):
    prompt = 'Suponha que você está trabalhando para identificar notícias falsas a partir de um site de checagem. Seu objetivo é, dado texto do site de checagem, gerar a possível notícia falsa que foi divulgada. Seja direto, respondendo somente a notícia falsa em questão, sem justificativas ou textos excedentes. Sabendo disso, identifique a notícia falsa que foi divulgada a partir do seguinte texto do site de checagem: ' + news

    response: ChatResponse = chat(model='olmo2', messages=[
        {
        'role': 'user',
        'content': prompt
        }
    ])

    return response.message.content

df['Model Answer'] = df['news'].apply(query_olmo2)

df.to_csv('output.tsv', sep = '\t')