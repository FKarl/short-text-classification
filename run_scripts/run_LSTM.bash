#!/usr/bin/env bash
lr=0.002
batch_size=32
epochs=100
layers=2
hidden_size=128
echo "Running LSTMs for all datasets"
echo "=================="
for dataset in "MR" "R8" "SearchSnippets" "Twitter" "TREC" "SST2" "NICE" "NICE2" "STOPS" "STOPS2"
do
    echo "Running $dataset on LSTM..."
    CUDA_VISIBLE_DEVICES=$1 python3 main.py $dataset LSTM --learning_rate=$lr --batch_size=$batch_size --num_train_epochs=$epochs --num_layers=$layers --hidden_size=$hidden_size
    echo "Running $dataset on Bi-LSTM..."
    CUDA_VISIBLE_DEVICES=$1 python3 main.py $dataset LSTM --learning_rate=$lr --batch_size=$batch_size --num_train_epochs=$epochs --num_layers=$layers --hidden_size=$hidden_size --bidirectional
    echo "Running $dataset on LSTM (GloVe)..."
    CUDA_VISIBLE_DEVICES=$1 python3 main.py $dataset LSTM --learning_rate=$lr --batch_size=$batch_size --num_train_epochs=$epochs --num_layers=$layers --hidden_size=$hidden_size --pretrained_embedding_path="glove/glove.6B.300d.txt"
    echo "Running $dataset on Bi-LSTM (GloVe)..."
    CUDA_VISIBLE_DEVICES=$1 python3 main.py $dataset LSTM --learning_rate=$lr --batch_size=$batch_size --num_train_epochs=$epochs --num_layers=$layers --hidden_size=$hidden_size --pretrained_embedding_path="glove/glove.6B.300d.txt" --bidirectional
done