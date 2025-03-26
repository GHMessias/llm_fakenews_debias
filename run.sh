#!/bin/bash
while getopts "j:" opt; do
    case $opt in
        j) JSON_PATH=$OPTARG ;;
        \?) echo "uso: run.sh -j json path"; exit 1 ;; 
    esac
done
    
ACTUAL_DATE=$(date +"%Y-%m-%d_%H:%M:%S")
echo "process start at $ACTUAL_DATE"

jq --arg date "$ACTUAL_DATE" '. + {"actual_date": $date}' "$JSON_PATH" > temp.json && mv temp.json "$JSON_PATH"

MAIN_FILE_PATH=$(pwd)

mkdir "results/$ACTUAL_DATE"
sleep 2

echo "Running LLM debias and summarization..."
python3 ollama_news_debias.py --config $JSON_PATH

# Escrevendo o caminho resultado no json
LLM_PROCESSED_PATH=results/$ACTUAL_DATE/llm_processed_data.tsv
echo "LLM debias/summarization completed. File saved at $LLM_PROCESSED_PATH"
jq --arg input_debiased_data_path "$LLM_PROCESSED_PATH" '. + {"input_debiased_data_path": $input_debiased_data_path}' "$JSON_PATH" > temp.json && mv temp.json "$JSON_PATH"

echo "Running data analysis..."
python3 dataset_analysis.py --config $JSON_PATH 
echo "Data analysis saved at COLOCAR PATH"

EMBEDDING_MODEL=$(jq -r '.embedding_model' "$JSON_PATH")
echo "Creating data embeddings using $EMBEDDING_MODEL"
python3 text_embedding.py --config $JSON_PATH

# Graph Generator
EMBEDDING_ORIGINAL_PATH=results/$ACTUAL_DATE/embedded_original_data.npy
EMBEDDING_DEBIASED_PATH=results/$ACTUAL_DATE/embedded_debiased_data.npy

jq --arg embedding_original_path "$EMBEDDING_ORIGINAL_PATH" '. + {"embedding_original_path": $embedding_original_path}' "$JSON_PATH" > temp.json && mv temp.json "$JSON_PATH"
jq --arg embedding_debiased_path "$EMBEDDING_DEBIASED_PATH" '. + {"embedding_debiased_path": $embedding_debiased_path}' "$JSON_PATH" > temp.json && mv temp.json "$JSON_PATH"

mkdir results/$ACTUAL_DATE/original_graphs
mkdir results/$ACTUAL_DATE/debiased_graphs
sleep 2

echo "generating graphs..."
python3 graph_generator.py --config $JSON_PATH
echo "all graphs successfully generated"

mkdir results/$ACTUAL_DATE/debiased_graphs/samples
mkdir results/$ACTUAL_DATE/original_graphs/samples

sleep 2
echo "processing data to pytorch_geometric format"
python3 data_processing.py --config $JSON_PATH
echo "data processed"

# execução dos experimentos
echo "starting experiments..."
python3 benchmark.py --config $JSON_PATH