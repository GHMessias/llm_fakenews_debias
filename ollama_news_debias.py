import pandas as pd
from ollama import chat
from ollama import ChatResponse
import utils.utils as utils
from tqdm import tqdm
import re

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
     df = df.sample(n = args.number_of_samples, random_state= args.seed)
     

def query_olmo2(news):
    # prompt = args.input_instruction + news

    response: ChatResponse = chat(model=args.llm_model, messages=[
         {'role' : 'user',
          'content' : args.input_instruction
         },
        {
        'role': 'user',
        'content': news
        }
    ])

    return response.message.content.replace('\n', ' ')

def split_think_model(text):
    return text.split('</think')[0]

def split_model_answer(text):
    return text.split('</think')[1]

# if específico para modelos da deep_seek
if 'deepseek' in args.llm_model:
    df['aux_column'] = df.progress_apply(lambda row: query_olmo2(row['news']) if row['label'] == 1 else None, axis=1)
    #  df[['think', 'model answer']] = df['aux_column'].apply(split_think_model)
    df['think'] = df['aux_column'].apply(split_think_model)
    df['Model Answer'] = df['aux_column'].apply(split_model_answer)
    df[['Model Answer','think', 'news']].to_csv(args.output_path, sep = '\t')
else:
    df['Model Answer'] = df.progress_apply(lambda row: query_olmo2(row['news']) if row['label'] == 1 else None, axis=1)
    df[['Model Answer', 'news']].to_csv(args.output_path, sep = '\t')