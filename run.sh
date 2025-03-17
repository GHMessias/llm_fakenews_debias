#!/bin/bash

echo "Executando processamento LLM..."
LLM_OUTPUT=$(python3 ollama_news_debias.py --config json_inputs/input_gemma_factcheckednews_1.json --number_of_samples 10)
# OUTPUT_FILE=$(python3 ollama_news_debias.py --config json_inputs/input_gemma_factcheckednews_1.json --number_of_samples 10 | grep "OUTPUT_FILE: " | cut -d ' ' -f2)  # Captura a saída do script

OUTPUT_FILE=$(echo "$LLM_OUTPUT" | grep "OUTPUT_FILE:" | cut -d ' ' -f2)
INPUT_PATH=$(echo "$LLM_OUTPUT" | grep "INPUT_FILE:" | cut -d ' ' -f2)


echo "------"
echo "Debiased file path: $OUTPUT_FILE"
echo "Original file path: $INPUT_PATH"

echo "Executando análise de dados..."
python3 dataset_analysis.py --input_debiased_data_path "$OUTPUT_FILE" --input_data_path "$INPUT_PATH"