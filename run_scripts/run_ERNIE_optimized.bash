#!/usr/bin/env bash
lr=25e-6
echo "Running ERNIE for all datasets"
echo "=================="
for dataset in "MR" "SearchSnippets" "Twitter" "TREC" "SST2" "NICE" "NICE2" "STOPS" "STOPS2"
do
    echo "Running $dataset..."
    CUDA_VISIBLE_DEVICES=$1 python3 main.py $dataset ERNIE --learning_rate=$lr
done
echo "Running R8..."
CUDA_VISIBLE_DEVICES=$1 python3 main.py $dataset ERNIE --learning_rate=$lr --batch_size=32