#!/usr/bin/env bash
echo "Running $2 for all"
echo "=================="
echo "Running BERT..."
CUDA_VISIBLE_DEVICES=$1 python3 main.py "$2" BERT --learning_rate=5e-5 --batch_size=128 --num_train_epochs=10
echo "Running ROBERTA..."
CUDA_VISIBLE_DEVICES=$1 python3 main.py "$2" ROBERTA --learning_rate=4e-5 --batch_size=128 --num_train_epochs=10
echo "Running DEBERTA..."
CUDA_VISIBLE_DEVICES=$1 python3 main.py "$2" DEBERTA --learning_rate=2e-5 --batch_size=128 --num_train_epochs=10
echo "Running ERNIE..."
CUDA_VISIBLE_DEVICES=$1 python3 main.py "$2" ERNIE --learning_rate=2e-5 --batch_size=256 --num_train_epochs=4
echo "Running ERNIE (optimized)..."
CUDA_VISIBLE_DEVICES=$1 python3 main.py "$2" ERNIE --learning_rate=25e-6
echo "Running DISTILBERT..."
CUDA_VISIBLE_DEVICES=$1 python3 main.py "$2" DISTILBERT  --learning_rate=5e-5 --batch_size=128 --num_train_epochs=10
echo "Running ALBERT..."
CUDA_VISIBLE_DEVICES=$1 python3 main.py "$2" ALBERT
echo "Running MLP..."
CUDA_VISIBLE_DEVICES=$1 python3 main.py "$2" MLP --learning_rate=1e-3 --batch_size=16 --num_train_epochs=100
echo "Running LSTM..."
CUDA_VISIBLE_DEVICES=$1 python3 main.py "$2" LSTM --learning_rate=0.002 --batch_size=32 --num_train_epochs=100 --num_layers=2 --hidden_size=128
echo "Running Bi-LSTM..."
CUDA_VISIBLE_DEVICES=$1 python3 main.py "$2" LSTM --learning_rate=0.002 --batch_size=32 --num_train_epochs=100 --num_layers=2 --hidden_size=128 --bidirectional
echo "Running LSTM (GLOVE)..."
CUDA_VISIBLE_DEVICES=$1 python3 main.py "$2" LSTM --learning_rate=0.002 --batch_size=32 --num_train_epochs=100 --num_layers=2 --hidden_size=128 --pretrained_embedding_path="glove/glove.6B.300d.txt"
echo "Running Bi-LSTM (GLOVE)..."
CUDA_VISIBLE_DEVICES=$1 python3 main.py "$2" LSTM --learning_rate=0.002 --batch_size=32 --num_train_epochs=100 --num_layers=2 --hidden_size=128 --pretrained_embedding_path="glove/glove.6B.300d.txt" --bidirectional