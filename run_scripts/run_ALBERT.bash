#!/usr/bin/env bash
lr=1e-5
batch_size=32
dropout=0
epochs=10
echo "Running ALBERT for all datasets"
echo "=================="
for dataset in "MR" "R8" "SearchSnippets" "Twitter" "TREC" "SST2" "NICE" "NICE2" "STOPS" "STOPS2"
do
    echo "Running $dataset..."
    CUDA_VISIBLE_DEVICES=$1 python3 main.py $dataset ALBERT --learning_rate=$lr --batch_size=$batch_size --num_train_epochs=$epochs --dropout=$dropout
done