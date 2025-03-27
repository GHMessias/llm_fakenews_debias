import pandas as pd
from ollama import chat
from ollama import ChatResponse
import utils.utils as utils
from tqdm import tqdm
import re
from datetime import datetime

tqdm.pandas()

args = utils.parse_arguments()
if args.config:
        config_params = utils.load_config_from_json(args.config)
        # Atualiza os parâmetros do argparse com os valores do JSON
        for key, value in config_params.items():
            setattr(args, key, value)

df = pd.read_csv(args.input_data_path, sep = '\t')
if args.number_of_samples:
     df = df.sample(n = args.number_of_samples, random_state=args.seed)
     

# TODO: organizar o number of samples para pegar os dados de ambas as classes
df_fake = df[df['label'] == 1]
df_true = df[df['label'] == -1]


def llm_query(news, prompt_path):
    print('processing text')
    prompt = load_markdown_prompt(prompt_path)

    response: ChatResponse = chat(model=args.llm_model, messages=[
         {'role' : 'user',
          'content' : prompt
         },
        {
        'role': 'user',
        'content': news
        }
    ],
    options = {"temperature" : args.temperature}
    )

    return response.message.content.replace('\n', ' ')

def split_think_model(text):
    return text.split('</think')[0]

def split_model_answer(text):
    return text.split('</think')[1]

def load_markdown_prompt(file_path):
    """Lê um arquivo Markdown e retorna seu conteúdo como string"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

date = args.actual_date
print(f"INPUT_FILE: {args.input_data_path}")
print(f"OUTPUT_FILE: {f'results/{date}/llm_processed_data.tsv'}")
print(f"LLM_MODEL: {args.llm_model}")

# if específico para modelos da deep_seek
# TODO: organizar para modelos do deepseek
if 'deepseek' in args.llm_model:
    # df['aux_column'] = df.progress_apply(lambda row: llm_query(row['news']) if row['label'] == 1 else None, axis=1)
    # #  df[['think', 'model answer']] = df['aux_column'].apply(split_think_model)
    # df['think'] = df['aux_column'].apply(split_think_model)
    # df['Model Answer'] = df['aux_column'].apply(split_model_answer)
    # df[['Model Answer','think', 'news']].to_csv(f'results/{date}/llm_processed_data.tsv', sep = '\t')
    print('NOT IMPLEMENTED ERROR')
else:
    df_fake['Model Answer'] = df_fake.progress_apply(lambda row: llm_query(row['news'], args.prompt_debias_input_path), axis=1)
    df_true['Model Answer'] = df_true.progress_apply(lambda row: llm_query(row['news'], args.prompt_summarization_input_path), axis=1)

    df_final = pd.concat([df_fake, df_true], ignore_index = True)
    df_final[['Model Answer', 'news', 'label']].to_csv(f'results/{date}/llm_processed_data.tsv', sep = '\t')

