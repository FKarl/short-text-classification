#!/usr/bin/env bash
lr=1e-3
batch_size=16
epochs=100
layers=1
hidden_size=1024
echo "Running WideMLP for all datasets"
echo "=================="
for dataset in "MR" "R8" "SearchSnippets" "Twitter" "TREC" "SST2" "NICE" "NICE2" "STOPS" "STOPS2"
do
    echo "Running $dataset..."
    CUDA_VISIBLE_DEVICES=$1 python3 main.py $dataset MLP --learning_rate=$lr --batch_size=$batch_size --num_train_epochs=$epochs --num_layers=$layers --hidden_size=$hidden_size
done