#!/usr/bin/env bash
lr=5e-5
batch_size=128
epochs=10
echo "Running BERT for all datasets"
echo "=================="
for dataset in "MR" "SearchSnippets" "Twitter" "TREC" "SST2" "NICE" "NICE2" "STOPS" "STOPS2"
do
    echo "Running $dataset..."
    CUDA_VISIBLE_DEVICES=$1 python3 main.py $dataset BERT --learning_rate=$lr --batch_size=$batch_size --num_train_epochs=$epochs
done
echo "Running R8..."
CUDA_VISIBLE_DEVICES=$1 python3 main.py $dataset BERT --learning_rate=$lr --batch_size=$batch_size --num_train_epochs=$epochs --batch_size=32