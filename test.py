import pandas as pd
import yake
from itertools import combinations
from torch_geometric.data import Data
import torch

def extract_keywords(text, ngram_range=(1, 3), top_k=20):
    # Extrai n-gramas entre unigramas e trigramas
    custom_kw_extractor = yake.KeywordExtractor(
        lan="pt",  # ou "en" se os textos forem em inglês
        n=ngram_range[1],  # extrai até trigramas
        top=top_k,
        features=None
    )
    keywords = custom_kw_extractor.extract_keywords(text)
    # Filtra por tamanho de n-grama desejado (1 a 3)
    selected = [kw for kw, score in keywords if len(kw.split()) in range(ngram_range[0], ngram_range[1]+1)]
    return set(selected)

def build_graph(df, text_column="news"):
    # Passo 1: extrair n-gramas relevantes de cada texto
    keyword_sets = df[text_column].apply(lambda txt: extract_keywords(txt)).tolist()
    
    # Passo 2: comparar cada par de textos
    edges = []
    for i, j in combinations(range(len(keyword_sets)), 2):
        if keyword_sets[i] & keyword_sets[j]:  # interseção não vazia
            edges.append((i, j))
            edges.append((j, i))  # grafo não direcionado (duplicar)

    if not edges:
        raise ValueError("Nenhuma conexão entre os textos encontrada com os critérios definidos.")
    
    # Passo 3: transformar em edge_index (formato [2, num_edges])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    # Passo 4: criar Data (PyG)
    data = Data(edge_index=edge_index, num_nodes=len(df))
    return data

# Exemplo de dataframe
df = pd.DataFrame({
    "news": [
        "Resíduos industriais geram poluição nas regiões urbanas.",
        "A gestão ambiental é essencial para resíduos industriais.",
        "A biodiversidade do cerrado sofre com o impacto ambiental."
    ]
})

graph_data = build_graph(df)
print(graph_data)
