#!/bin/bash

USE_JSON_DATE=false
USE_JSON_LLM_PATH=false

# Parsing manual de flags simples
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        -j)
            JSON_PATH="$2"
            shift 2
            ;;
        -cd)
            USE_JSON_DATE=true
            shift
            ;;
        -cp)
            USE_JSON_LLM_PATH=true
            shift
            ;;
        -ceo)
            CUSTOM_EMBEDDING_ORIGINAL="$2"
            shift 2
            ;;
        -ced)
            CUSTOM_EMBEDDING_DEBIASED="$2"
            shift 2
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# Restaura argumentos posicionais, se necessário
set -- "${POSITIONAL_ARGS[@]}"

# Verifica JSON path
if [ -z "$JSON_PATH" ]; then
    echo "Uso: ./run.sh -j json_path [-cd] [-cp] [-ceo caminho_embedding_original] [-ced caminho_embedding_debiased]"
    exit 1
fi

# Define ACTUAL_DATE
if $USE_JSON_DATE; then
    ACTUAL_DATE=$(jq -r '.actual_date' "$JSON_PATH")
    echo "Using the date $ACTUAL_DATE previously provided"
    mkdir -p "results/$ACTUAL_DATE/dataset_analysis"
    sleep 2
    if [ "$ACTUAL_DATE" == "null" ]; then
        echo "Erro: 'actual_date' não encontrado no JSON." >&2
        exit 1
    fi
else
    ACTUAL_DATE=$(date +"%Y-%m-%d_%H:%M:%S")
    echo "Using new date $ACTUAL_DATE"
    jq --arg date "$ACTUAL_DATE" '. + {"actual_date": $date}' "$JSON_PATH" > temp.json && mv temp.json "$JSON_PATH"
    mkdir -p "results/$ACTUAL_DATE/dataset_analysis"
    sleep 2
fi

# Define LLM_PROCESSED_PATH
if $USE_JSON_LLM_PATH; then
    LLM_PROCESSED_PATH=$(jq -r '.input_debiased_data_path' "$JSON_PATH")
    echo "Using the processed path $LLM_PROCESSED_PATH previously provided"
    if [ "$LLM_PROCESSED_PATH" == "null" ]; then
        echo "Erro: 'input_debiased_data_path' não encontrado no JSON." >&2
        exit 1
    fi
else
    echo "Running LLM debias and summarization..."
    python3 ollama_news_debias.py --config $JSON_PATH
    LLM_PROCESSED_PATH=results/$ACTUAL_DATE/llm_processed_data.tsv
    echo "Saving new processed path at $LLM_PROCESSED_PATH"
    jq --arg input_debiased_data_path "$LLM_PROCESSED_PATH" '. + {"input_debiased_data_path": $input_debiased_data_path}' "$JSON_PATH" > temp.json && mv temp.json "$JSON_PATH"
    echo "LLM debias/summarization completed. File saved at $LLM_PROCESSED_PATH"
fi

# Caso os caminhos dos embeddings sejam fornecidos diretamente
if [ -n "$CUSTOM_EMBEDDING_ORIGINAL" ]; then
    echo "Usando caminho de embedding original fornecido: $CUSTOM_EMBEDDING_ORIGINAL"
    jq --arg embedding_original_path "$CUSTOM_EMBEDDING_ORIGINAL" '. + {"embedding_original_path": $embedding_original_path}' "$JSON_PATH" > temp.json && mv temp.json "$JSON_PATH"
fi

if [ -n "$CUSTOM_EMBEDDING_DEBIASED" ]; then
    echo "Usando caminho de embedding debiased fornecido: $CUSTOM_EMBEDDING_DEBIASED"
    jq --arg embedding_debiased_path "$CUSTOM_EMBEDDING_DEBIASED" '. + {"embedding_debiased_path": $embedding_debiased_path}' "$JSON_PATH" > temp.json && mv temp.json "$JSON_PATH"
fi

# Geração de embeddings, se necessário
if [ -z "$CUSTOM_EMBEDDING_ORIGINAL" ] || [ -z "$CUSTOM_EMBEDDING_DEBIASED" ]; then
    EMBEDDING_MODEL=$(jq -r '.embedding_model' "$JSON_PATH")
    echo "Creating data embeddings using $EMBEDDING_MODEL"
    python3 text_embedding.py --config $JSON_PATH

    EMBEDDING_ORIGINAL_PATH=results/$ACTUAL_DATE/embedded_original_data.npy
    EMBEDDING_DEBIASED_PATH=results/$ACTUAL_DATE/embedded_debiased_data.npy

    jq --arg embedding_original_path "$EMBEDDING_ORIGINAL_PATH" '. + {"embedding_original_path": $embedding_original_path}' "$JSON_PATH" > temp.json && mv temp.json "$JSON_PATH"
    jq --arg embedding_debiased_path "$EMBEDDING_DEBIASED_PATH" '. + {"embedding_debiased_path": $embedding_debiased_path}' "$JSON_PATH" > temp.json && mv temp.json "$JSON_PATH"
else
    echo "Embedding paths were provided via flags. Skipping embedding generation."
fi

echo "Running data analysis..."
python3 dataset_analysis.py --config $JSON_PATH
echo "Data analysis saved at COLOCAR PATH"

mkdir -p results/$ACTUAL_DATE/original_graphs/samples
mkdir -p results/$ACTUAL_DATE/debiased_graphs/samples
sleep 2

echo "Generating graphs..."
python3 graph_generator.py --config $JSON_PATH
echo "All graphs successfully generated"

echo "Processing data to pytorch_geometric format..."
python3 data_processing.py --config $JSON_PATH
echo "Data processed"

echo "Starting experiments..."
mkdir -p results/$ACTUAL_DATE/benchmark_outputs/debiased
mkdir -p results/$ACTUAL_DATE/benchmark_outputs/original
python3 benchmark.py --config $JSON_PATH
