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

df = df[df['label'] == 1]
if args.number_of_samples:
     df = df.sample(n = args.number_of_samples, random_state=args.seed)
     

def query_olmo2(news):
    prompt = load_markdown_prompt(args.prompt_input_path)

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

now_date = datetime.now().strftime(("%Y-%m-%d_%H:%M:%S"))
output_path = args.output_path + '_' + now_date
print(f"INPUT_FILE: {args.input_data_path}")
print(f"OUTPUT_FILE: {output_path}")
print(f"LLM_MODEL: {args.llm_model}")

# if específico para modelos da deep_seek
if 'deepseek' in args.llm_model:
    df['aux_column'] = df.progress_apply(lambda row: query_olmo2(row['news']) if row['label'] == 1 else None, axis=1)
    #  df[['think', 'model answer']] = df['aux_column'].apply(split_think_model)
    df['think'] = df['aux_column'].apply(split_think_model)
    df['Model Answer'] = df['aux_column'].apply(split_model_answer)
    df[['Model Answer','think', 'news']].to_csv(output_path, sep = '\t')
else:
    df['Model Answer'] = df.progress_apply(lambda row: query_olmo2(row['news']) if row['label'] == 1 else None, axis=1)
    df[['Model Answer', 'news']].to_csv(output_path, sep = '\t')

