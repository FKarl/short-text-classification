"""
Description: Run text classification on the given dataset with the given model.
"""
import json
import logging
import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import get_linear_schedule_with_warmup, AdamW, AutoModelForSequenceClassification, Trainer, \
    TrainingArguments, AutoTokenizer

from ensemble_models import SimplifiedStacking, SimplifiedWeightedBoost, WeightedEnsemble
from data import Dataset, SimpleDataset, load_data, load_pretrained_embeddings, build_tokenizer_for_word_embeddings, \
    prepare_data_custom_tokenizer, prepare_data
from models import MLP, collate_for_mlp, LSTM, collate_for_lstm

try:
    import wandb

    WANDB = True
except ImportError:
    logging.info("Wandb not installed. Skipping tracking.")
    WANDB = False

MODELS = {
    "BERT": "bert-base-uncased",
    "ROBERTA": "roberta-base",
    "DEBERTA": "microsoft/deberta-base",
    "MLP": "bert-base-uncased",
    "ERNIE": "nghuyong/ernie-2.0-base-en",
    "DISTILBERT": "distilbert-base-uncased",
    "ALBERT": "albert-base-v2",
    "LSTM": "bert-base-uncased",
}

VALID_DATASETS = ['MR', 'R8', 'SearchSnippets', 'Twitter', 'TREC', 'SST2', 'NICE', 'NICE2', 'STOPS', 'STOPS2']
VALID_MODELS = list(MODELS.keys()) + ["STACKING", "WEIGHTED_BOOST", "WEIGHTED"]


def compute_metrics(pred):
    """
    Compute the metrics for the given predictions.
    :param pred: the predictions
    :return: accuracy
    """

    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}


def train_transformer(model, dataset, output_dir, training_batch_size, eval_batch_size, learning_rate,
                      num_train_epochs, weight_decay,
                      disable_tqdm=False):
    """
    Train and fine-tune the model using HuggingFace's PyTorch implementation and the Trainer API.
    """

    # training params
    model_ckpt = MODELS[model]
    print(model_ckpt)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output = f"{output_dir}/{model_ckpt}-finetuned-{dataset['name']}"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    # max length of 512 for ERNIE as it is not predefined in the model
    if model == "ERNIE":
        test_data, train_data, label_dict = prepare_data(dataset, tokenizer, Dataset, max_length=512)
    else:
        test_data, train_data, label_dict = prepare_data(dataset, tokenizer, Dataset)

    model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=len(label_dict)).to(device)
    logging_steps = len(train_data) // training_batch_size

    # train
    if WANDB:
        wandb.watch(model)
        training_args = TrainingArguments(output_dir=output,
                                          num_train_epochs=num_train_epochs,
                                          learning_rate=learning_rate,
                                          per_device_train_batch_size=training_batch_size,
                                          per_device_eval_batch_size=eval_batch_size,
                                          weight_decay=weight_decay,
                                          evaluation_strategy="epoch",
                                          disable_tqdm=disable_tqdm,
                                          logging_steps=logging_steps,
                                          log_level="error",
                                          logging_dir="./logs",
                                          report_to="wandb")
    else:
        training_args = TrainingArguments(output_dir=output,
                                          num_train_epochs=num_train_epochs,
                                          learning_rate=learning_rate,
                                          per_device_train_batch_size=training_batch_size,
                                          per_device_eval_batch_size=eval_batch_size,
                                          weight_decay=weight_decay,
                                          evaluation_strategy="epoch",
                                          disable_tqdm=disable_tqdm,
                                          logging_steps=logging_steps,
                                          log_level="error",
                                          logging_dir="./logs")

    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=train_data,
                      eval_dataset=test_data,
                      compute_metrics=compute_metrics,
                      tokenizer=tokenizer)

    trainer.train()

    evaluate_trainer(trainer, test_data, output_dir)

    # save model
    model.save_pretrained(f"{output}/model")


def evaluate_trainer(trainer, test_data, output_dir):
    """
    Evaluate the fine-tuned trainer on the test set.
    Therefore, the accuracy is computed and a confusion matrix is generated.
    """

    # accuracy
    prediction_output = trainer.predict(test_data)
    logging.info(f"Prediction metrics: {prediction_output.metrics}")

    # confusion matrix
    y_preds = np.argmax(prediction_output.predictions, axis=1)
    y_true = prediction_output.label_ids
    cm = confusion_matrix(y_true, y_preds)
    logging.info(f"Confusion matrix:\n{cm}")

    # create file if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # save results to file
    with open(f"{output_dir}/eval_results.json", "a") as f:
        f.write("\n")
        json.dump(prediction_output.metrics, f)

    if WANDB:
        wandb.log(prediction_output.metrics)


def train_mlp(dataset, output_dir, epochs, warmup_steps, learning_rate, weight_decay, gradient_accumulation_steps,
              training_batch_size, eval_batch_size, dropout, hidden_size, num_hidden_layers, pretrained_embedding_path,
              freeze_embedding):
    """
    Train a simple MLP model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output = f"{output_dir}/mlp-finetuned-{dataset['name']}"

    if (pretrained_embedding_path is not None) and (pretrained_embedding_path != ""):
        logging.info("Loading pretrained embedding")

        vocab, embedding = load_pretrained_embeddings(pretrained_embedding_path, unk_token="[UNK]")
        logging.debug(f"Vocab size: {len(vocab)}")
        logging.debug(f"Embedding size: {embedding.shape}")
        tokenizer = build_tokenizer_for_word_embeddings(vocab, "[UNK]")
        test_data, train_data, label_dict = prepare_data_custom_tokenizer(dataset, tokenizer, SimpleDataset)
        vocab_size = len(vocab)
    else:
        logging.info("Loading tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(MODELS["MLP"])

        test_data, train_data, label_dict = prepare_data(dataset, tokenizer, SimpleDataset)
        vocab_size = tokenizer.vocab_size
        embedding = None

    train_loader = DataLoader(train_data,
                              collate_fn=collate_for_mlp,
                              shuffle=True,
                              batch_size=training_batch_size)

    model = MLP(vocab_size,
                len(label_dict),
                dropout=dropout,
                hidden_size=hidden_size,
                num_hidden_layers=num_hidden_layers,
                pretrained_embedding=embedding,
                freeze=freeze_embedding
                ).to(device)

    train_model(device, epochs, eval_batch_size, gradient_accumulation_steps, learning_rate, model, test_data,
                train_data, train_loader, warmup_steps, weight_decay, output)


def train_lstm(dataset, output_dir, epochs, warmup_steps, learning_rate, weight_decay, gradient_accumulation_steps,
               training_batch_size, eval_batch_size, bidirectional, dropout, num_layers, hidden_size,
               pretrained_embedding_path, freeze_embedding):
    """
    Train a simple LSTM model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output = f"{output_dir}/lstm-finetuned-{dataset['name']}"

    if (pretrained_embedding_path is not None) and (pretrained_embedding_path != ""):
        logging.info("Loading pretrained embedding")

        vocab, embedding = load_pretrained_embeddings(pretrained_embedding_path, unk_token="[UNK]")
        logging.debug(f"Vocab size: {len(vocab)}")
        logging.debug(f"Embedding size: {embedding.shape}")
        tokenizer = build_tokenizer_for_word_embeddings(vocab, "[UNK]")
        test_data, train_data, label_dict = prepare_data_custom_tokenizer(dataset, tokenizer, SimpleDataset)
        vocab_size = len(vocab)
    else:
        logging.info("Loading tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(MODELS["LSTM"])

        test_data, train_data, label_dict = prepare_data(dataset, tokenizer, SimpleDataset)
        vocab_size = tokenizer.vocab_size
        embedding = None

    # change collate?
    train_loader = DataLoader(train_data,
                              collate_fn=collate_for_lstm,
                              shuffle=True,
                              batch_size=training_batch_size)

    model = LSTM(vocab_size,
                 len(label_dict),
                 bidirectional=bidirectional,
                 dropout=dropout,
                 num_layers=num_layers,
                 hidden_size=hidden_size,
                 pretrained_embedding=embedding,
                 freeze=freeze_embedding
                 ).to(device)

    train_model(device, epochs, eval_batch_size, gradient_accumulation_steps, learning_rate, model, test_data,
                train_data, train_loader, warmup_steps, weight_decay, output)


def train_model(device, epochs, eval_batch_size, gradient_accumulation_steps, learning_rate, model, test_data,
                train_data, train_loader, warmup_steps, weight_decay, output_dir):
    """
    Train a PyTorch model with the specified parameters.
    Used by both the MLP and LSTM model.
    """
    if WANDB:
        wandb.watch(model)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    t_total = len(train_loader) // gradient_accumulation_steps * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=t_total)
    logging.info(f"Training model on {device}")
    global_step = 0
    training_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(epochs, desc="Epoch")

    for epoch in train_iterator:
        epoch_iterator = tqdm(train_loader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(device) for t in batch)

            flat_docs, offsets, labels = batch
            outputs = model(flat_docs, offsets, labels)

            loss = outputs[0]
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            loss.backward()
            training_loss += loss.item()
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1
                if WANDB:
                    wandb.log({
                        "train/epoch": epoch,
                        "train/train_loss": loss,
                        "train/learning_rate": scheduler.get_lr()[0]
                    })

            logging_steps = len(train_data) // 16
            if global_step % logging_steps == 0:
                acc, eval_loss = evaluate_model(model, test_data, eval_batch_size, device, output_dir)
                logging.info(f"Epoch {epoch} Step {step} Loss {loss.item()} Eval loss {eval_loss} Acc {acc}")

    # eval
    acc, eval_loss = evaluate_model(model, test_data, eval_batch_size, device, output_dir)
    logging.info(f"Evaluation loss: {eval_loss}")
    logging.info(f"Evaluation accuracy: {acc}")


def evaluate_model(model, test_data, eval_batch_size, device, output_dir):
    """
    Evaluate a trained PyTorch model.
    Therefore, the accuracy and loss are calculated.
    """
    data_loader = DataLoader(test_data,
                             collate_fn=collate_for_mlp,
                             shuffle=False,
                             batch_size=eval_batch_size)
    all_logits = []
    all_targets = []
    eval_steps, eval_loss = 0, 0.0
    for batch in tqdm(data_loader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            flat_inputs, lengths, labels = batch
            outputs = model(flat_inputs, lengths, labels)
            all_targets.append(labels.detach().cpu())

        eval_steps += 1
        loss, logits = outputs[:2]
        eval_loss += loss.mean().item()
        all_logits.append(logits.detach().cpu())

    logits = torch.cat(all_logits).numpy()
    targets = torch.cat(all_targets).numpy()
    eval_loss /= eval_steps
    preds = np.argmax(logits, axis=1)
    acc = (preds == targets).sum() / targets.size

    # create file if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # append result to file
    with open(f"{output_dir}/eval_results.json", "a") as f:
        f.write("\n")
        json.dump({"acc": acc, "model parameter": str(model)}, f)

    if WANDB:
        wandb.log({
            "eval/loss": eval_loss,
            "eval/accuracy": acc
        })

    return acc, eval_loss


def train_stacking(dataset, model1_name, model2_name, meta_model_name, m1_dropout, m2_dropout, m3_dropout,
                   m1_hidden_size, m2_hidden_size, m3_hidden_size, m1_num_layers, m2_num_layers, m3_num_layers,
                   batch_size, m1_lr, m2_lr, m3_lr, m1_weight_decay, m2_weight_decay, m3_weight_decay, epochs):
    """
    Train a stacking model.

    :param dataset: dataset
    :param model1_name: name of the first model
    :param model2_name: name of the second model
    :param meta_model_name: name of the meta model
    :param m1_dropout: dropout of the first model
    :param m2_dropout: dropout of the second model
    :param m3_dropout: dropout of the meta model
    :param m1_hidden_size: hidden size of the first model
    :param m2_hidden_size: hidden size of the second model
    :param m3_hidden_size: hidden size of the third model
    :param m1_num_layers: number of layers of the first model
    :param m2_num_layers: number of layers of the second model
    :param m3_num_layers: number of layers of the third model
    :param batch_size: batch size for all models
    :param m1_lr: learning rate of the first model
    :param m2_lr: learning rate of the second model
    :param m3_lr: learning rate of the meta model
    :param m1_weight_decay: weight decay of the first model
    :param m2_weight_decay: weight decay of the second model
    :param m3_weight_decay: weight decay of the meta model
    :param epochs: number of epochs for all models
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_dict = dataset['label_dict']

    # setup models
    is_m1_transformer, m1, tokenizer1 = setup_model(device, label_dict, m1_dropout, m1_hidden_size, m1_num_layers,
                                                    model1_name)
    is_m2_transformer, m2, tokenizer2 = setup_model(device, label_dict, m2_dropout, m2_hidden_size, m2_num_layers,
                                                    model2_name)
    is_m3_transformer, m3, tokenizer3 = setup_model(device, label_dict, m3_dropout, m3_hidden_size, m3_num_layers,
                                                    meta_model_name)

    model = SimplifiedStacking(m1, m2, m3, is_m1_transformer, is_m2_transformer, is_m3_transformer, tokenizer1,
                               tokenizer2, tokenizer3)
    model.fit(dataset, batch_size, m1_lr, m2_lr, m3_lr, m1_weight_decay, m2_weight_decay, m3_weight_decay, epochs,
              device, )
    acc = model.evaluate(dataset, batch_size, device)

    if WANDB:
        wandb.log({
            "eval/accuracy": acc
        })


def train_weighted_boost(dataset, model1_name, model2_name, m1_dropout, m2_dropout, m1_hidden_size, m2_hidden_size,
                         m1_num_layers, m2_num_layers, batch_size, m1_lr, m2_lr, m1_weight_decay, m2_weight_decay,
                         epochs, alpha):
    """
    Train a weighted boosting model.

    :param dataset: dataset
    :param model1_name: name of the first model
    :param model2_name: name of the second model
    :param m1_dropout: dropout of the first model
    :param m2_dropout: dropout of the second model
    :param m1_hidden_size: hidden size of the first model
    :param m2_hidden_size: hidden size of the second model
    :param m1_num_layers: number of layers of the first model
    :param m2_num_layers: number of layers of the second model
    :param batch_size: batch size for all models
    :param m1_lr: learning rate of the first model
    :param m2_lr: learning rate of the second model
    :param m1_weight_decay: weight decay of the first model
    :param m2_weight_decay: weight decay of the second model
    :param epochs: number of epochs
    :param alpha: weight of the first model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_dict = dataset['label_dict']

    # setup models
    is_m1_transformer, m1, tokenizer1 = setup_model(device, label_dict, m1_dropout, m1_hidden_size, m1_num_layers,
                                                    model1_name)
    is_m2_transformer, m2, tokenizer2 = setup_model(device, label_dict, m2_dropout, m2_hidden_size, m2_num_layers,
                                                    model2_name)

    model = SimplifiedWeightedBoost(m1, m2, is_m1_transformer, is_m2_transformer, tokenizer1, tokenizer2)
    model.fit(dataset, batch_size, m1_lr, m2_lr, m1_weight_decay, m2_weight_decay, epochs, device)

    acc = model.evaluate(dataset, batch_size, device, alpha)
    logging.info(f"alpha: {alpha}, acc: {acc}")
    eval_for_different_alpha(acc, batch_size, dataset, device, model, alpha)


def eval_for_different_alpha(init_acc, batch_size, dataset, device, model, alpha):
    """
    Evaluate the model for different alpha values.
    The values are chosen in a range from 0.1 to 0.9 with a step size of 0.1.

    :param init_acc: the initial accuracy
    :param batch_size: the batch size
    :param dataset: the dataset
    :param device: the device
    :param model: the model
    :param alpha: the initial alpha value
    """
    best_acc = init_acc
    # test different alpha values
    for a in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        acc = model.evaluate(dataset, batch_size, device, a)
        logging.info(f"alpha: {a}, acc: {acc}")
        if acc > best_acc:
            best_acc = acc
            alpha = a
    if WANDB:
        wandb.log({
            "eval/accuracy": acc,
            "best alpha": alpha,
            "eval/accuracy with fix alpha": init_acc
        })


def train_weighted(dataset, model1_name, model2_name, m1_dropout, m2_dropout, m1_hidden_size, m2_hidden_size,
                   m1_num_layers, m2_num_layers, batch_size, m1_lr, m2_lr, m1_weight_decay, m2_weight_decay,
                   m1_epochs, m2_epochs, alpha):
    """
    Train a weighted ensemble model

    :param dataset: dataset
    :param model1_name: name of the first model
    :param model2_name: name of the second model
    :param m1_dropout: dropout of the first model
    :param m2_dropout: dropout of the second model
    :param m1_hidden_size: hidden size of the first model
    :param m2_hidden_size: hidden size of the second model
    :param m1_num_layers: number of layers of the first model
    :param m2_num_layers: number of layers of the second model
    :param batch_size: batch size for all models
    :param m1_lr: learning rate of the first model
    :param m2_lr: learning rate of the second model
    :param m1_weight_decay: weight decay of the first model
    :param m2_weight_decay: weight decay of the second model
    :param m1_epochs: number of epochs of the first model
    :param m2_epochs: number of epochs of the second model
    :param alpha: weight of the first model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_dict = dataset['label_dict']

    # setup models
    is_m1_transformer, m1, tokenizer1 = setup_model(device, label_dict, m1_dropout, m1_hidden_size, m1_num_layers,
                                                    model1_name)
    is_m2_transformer, m2, tokenizer2 = setup_model(device, label_dict, m2_dropout, m2_hidden_size, m2_num_layers,
                                                    model2_name)

    model = WeightedEnsemble(m1, m2, is_m1_transformer, is_m2_transformer, tokenizer1, tokenizer2)
    model.fit(dataset, batch_size, m1_lr, m2_lr, m1_weight_decay, m2_weight_decay, m1_epochs, m2_epochs, device)

    acc = model.evaluate(dataset, batch_size, device, alpha)
    logging.info(f"alpha: {alpha}, acc: {acc}")
    eval_for_different_alpha(acc, batch_size, dataset, device, model, alpha)


def setup_model(device, label_dict, dropout, hidden_size, num_layers, model_name):
    """
    Set up a model for training based on the given model name.
    Used for the ensemble models.

    :param device: device to use for training
    :param label_dict: label dictionary
    :param dropout: dropout rate
    :param hidden_size: hidden size
    :param num_layers: number of layers
    :param model_name: name of the model
    :return: is_transformer, model, tokenizer
    """
    if model_name == "MLP":
        is_m1_transformer = False
        tokenizer1 = AutoTokenizer.from_pretrained(MODELS["MLP"])
        vocab_size = tokenizer1.vocab_size

        m1 = MLP(vocab_size,
                 len(label_dict),
                 dropout=dropout,
                 hidden_size=hidden_size,
                 num_hidden_layers=num_layers,
                 pretrained_embedding=None,
                 freeze=False
                 ).to(device)
    elif model_name == "LSTM":
        is_m1_transformer = False
        tokenizer1 = AutoTokenizer.from_pretrained(MODELS["LSTM"])
        vocab_size = tokenizer1.vocab_size

        m1 = LSTM(vocab_size,
                  len(label_dict),
                  bidirectional=False,
                  dropout=dropout,
                  num_layers=num_layers,
                  hidden_size=hidden_size,
                  pretrained_embedding=None,
                  freeze=False
                  ).to(device)
    # Transformer:
    else:
        is_m1_transformer = True
        tokenizer1 = AutoTokenizer.from_pretrained(MODELS[model_name])

        m1 = AutoModelForSequenceClassification.from_pretrained(MODELS[model_name], num_labels=len(label_dict)).to(
            device)
    return is_m1_transformer, m1, tokenizer1


def main():
    """
    The main entry point of the script.
    Parses the arguments and starts the training.
    """
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run text classification on the given dataset with the given model.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # general arguments
    parser.add_argument('dataset', type=str, choices=VALID_DATASETS, help='Dataset to use.')
    parser.add_argument('model', type=str, choices=VALID_MODELS, help='Model to use.')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory.')
    parser.add_argument('--log_level', type=str, default='info', help='Log level.')
    parser.add_argument('--log_to_file', action='store_true', help='Log to file.')
    parser.add_argument('--log_file', type=str, default='log.txt', help='Log file.')

    # training arguments
    parser.add_argument('--batch_size', type=int, default=128, help='The batch size.')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate.')
    parser.add_argument('--num_train_epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--weight_decay', type=float, default=0.00, help='Weight decay.')
    parser.add_argument('--warmup_steps', type=int, default=0, help='Number of warmup steps.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of gradient accumulation steps.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout.')

    # lstm arguments
    parser.add_argument('--bidirectional', action='store_true', help='Use bidirectional LSTM.')

    # lstm / mlp arguments
    parser.add_argument('--num_layers', type=int, default=1, help='Number of layers.')
    parser.add_argument('--hidden_size', type=int, default=1024, help='Hidden size.')
    parser.add_argument('--pretrained_embedding_path', type=str, default=None, help='Path to pretrained embeddings.')
    parser.add_argument('--unfreeze_embedding', action='store_true', help='Unfreeze embedding.')

    # ensemble arguments
    parser.add_argument('--m1', type=str, choices=VALID_MODELS, help='Model 1 if ensemble is used.')
    parser.add_argument('--m2', type=str, choices=VALID_MODELS, help='Model 2 if ensemble is used.')
    parser.add_argument('--mm', type=str, choices=VALID_MODELS, help='Meta model if stacking ensemble is used.')

    parser.add_argument('--m1_learning_rate', type=float, default=5e-5, help='Learning rate.')
    parser.add_argument('--m1_weight_decay', type=float, default=0.00, help='Weight decay.')
    parser.add_argument('--m1_warmup_steps', type=int, default=0, help='Number of warmup steps.')
    parser.add_argument('--m1_dropout', type=float, default=0.5, help='Dropout.')
    parser.add_argument('--m1_num_layers', type=int, default=1, help='Number of layers.')
    parser.add_argument('--m1_hidden_size', type=int, default=1024, help='Hidden size.')

    parser.add_argument('--m2_learning_rate', type=float, default=5e-5, help='Learning rate.')
    parser.add_argument('--m2_weight_decay', type=float, default=0.00, help='Weight decay.')
    parser.add_argument('--m2_warmup_steps', type=int, default=0, help='Number of warmup steps.')
    parser.add_argument('--m2_dropout', type=float, default=0.5, help='Dropout.')
    parser.add_argument('--m2_num_layers', type=int, default=1, help='Number of layers.')
    parser.add_argument('--m2_hidden_size', type=int, default=1024, help='Hidden size.')
    parser.add_argument('--m2_num_train_epochs', type=int, default=10,
                        help='Number of training epochs. Only used with WEIGHTED')

    parser.add_argument('--mm_learning_rate', type=float, default=5e-5, help='Learning rate.')
    parser.add_argument('--mm_weight_decay', type=float, default=0.00, help='Weight decay.')
    parser.add_argument('--mm_warmup_steps', type=int, default=0, help='Number of warmup steps.')
    parser.add_argument('--mm_dropout', type=float, default=0.5, help='Dropout.')
    parser.add_argument('--mm_num_layers', type=int, default=1, help='Number of layers.')
    parser.add_argument('--mm_hidden_size', type=int, default=1024, help='Hidden size.')

    parser.add_argument('--alpha', type=float, default=0.5, help='Weight of model 1 in weighted ensemble.')

    args = parser.parse_args()

    # Set up logging
    log_level = getattr(logging, args.log_level.upper())
    if args.log_to_file:
        logging.basicConfig(filename=f'{args.output_dir}/{args.log_file}', level=log_level)
    else:
        logging.basicConfig(level=log_level)

    # init wandb
    if WANDB:
        # init wandb name
        model = args.model
        if model == 'LSTM' and args.bidirectional:
            name = 'Bi-LSTM'
        elif model == 'WEIGHTED_BOOST' or model == 'WEIGHTED':
            name = f'{model} ({args.m1} + {args.m2})'
        elif model == 'STACKING':
            name = f'{model} ({args.m1} + {args.m2} -> {args.mm})'
        else:
            name = model

        wandb.init(project='Bachelor-Thesis',
                   name=name,
                   config=vars(args))
        config = wandb.config
    else:
        config = vars(args)

    logging.info("Starting...")
    logging.debug("Arguments: %s", args)

    # Start training
    logging.info(f"Loading {args.dataset} data...")
    dataset = load_data(args.dataset)
    if args.model == "MLP":
        train_mlp(dataset,
                  args.output_dir,
                  eval_batch_size=config["batch_size"],
                  training_batch_size=config["batch_size"],
                  learning_rate=config["learning_rate"],
                  epochs=config["num_train_epochs"],
                  weight_decay=config["weight_decay"],
                  warmup_steps=config["warmup_steps"],
                  gradient_accumulation_steps=config["gradient_accumulation_steps"],
                  dropout=config["dropout"],
                  hidden_size=config["hidden_size"],
                  num_hidden_layers=config["num_layers"],
                  pretrained_embedding_path=config["pretrained_embedding_path"],
                  freeze_embedding=not config["unfreeze_embedding"],
                  )
    elif args.model == "LSTM":
        train_lstm(dataset,
                   config["output_dir"],
                   epochs=config["num_train_epochs"],
                   warmup_steps=config["warmup_steps"],
                   learning_rate=config["learning_rate"],
                   weight_decay=config["weight_decay"],
                   training_batch_size=config["batch_size"],
                   eval_batch_size=config["batch_size"],
                   bidirectional=config["bidirectional"],
                   num_layers=config["num_layers"],
                   hidden_size=config["hidden_size"],
                   dropout=config["dropout"],
                   gradient_accumulation_steps=config["gradient_accumulation_steps"],
                   pretrained_embedding_path=config["pretrained_embedding_path"],
                   freeze_embedding=not config["unfreeze_embedding"]
                   )
    elif args.model == "STACKING":
        train_stacking(dataset,
                       model1_name=config["m1"],
                       model2_name=config["m2"],
                       meta_model_name=config["mm"],
                       m1_dropout=config["m1_dropout"],
                       m2_dropout=config["m2_dropout"],
                       m3_dropout=config["mm_dropout"],
                       m1_hidden_size=config["m1_hidden_size"],
                       m2_hidden_size=config["m2_hidden_size"],
                       m3_hidden_size=config["mm_hidden_size"],
                       m1_num_layers=config["m1_num_layers"],
                       m2_num_layers=config["m2_num_layers"],
                       m3_num_layers=config["mm_num_layers"],
                       batch_size=config["batch_size"],
                       m1_lr=config["m1_learning_rate"],
                       m2_lr=config["m2_learning_rate"],
                       m3_lr=config["mm_learning_rate"],
                       m1_weight_decay=config["m1_weight_decay"],
                       m2_weight_decay=config["m2_weight_decay"],
                       m3_weight_decay=config["mm_weight_decay"],
                       epochs=config["num_train_epochs"]
                       )
    elif args.model == "WEIGHTED_BOOST":
        train_weighted_boost(dataset,
                             model1_name=config["m1"],
                             model2_name=config["m2"],
                             m1_dropout=config["m1_dropout"],
                             m2_dropout=config["m2_dropout"],
                             m1_hidden_size=config["m1_hidden_size"],
                             m2_hidden_size=config["m2_hidden_size"],
                             m1_num_layers=config["m1_num_layers"],
                             m2_num_layers=config["m2_num_layers"],
                             batch_size=config["batch_size"],
                             m1_lr=config["m1_learning_rate"],
                             m2_lr=config["m2_learning_rate"],
                             m1_weight_decay=config["m1_weight_decay"],
                             m2_weight_decay=config["m2_weight_decay"],
                             epochs=config["num_train_epochs"],
                             alpha=config["alpha"]
                             )
    elif args.model == "WEIGHTED":
        train_weighted(dataset,
                       model1_name=config["m1"],
                       model2_name=config["m2"],
                       m1_dropout=config["m1_dropout"],
                       m2_dropout=config["m2_dropout"],
                       m1_hidden_size=config["m1_hidden_size"],
                       m2_hidden_size=config["m2_hidden_size"],
                       m1_num_layers=config["m1_num_layers"],
                       m2_num_layers=config["m2_num_layers"],
                       batch_size=config["batch_size"],
                       m1_lr=config["m1_learning_rate"],
                       m2_lr=config["m2_learning_rate"],
                       m1_weight_decay=config["m1_weight_decay"],
                       m2_weight_decay=config["m2_weight_decay"],
                       m1_epochs=config["num_train_epochs"],
                       m2_epochs=config["m2_num_train_epochs"],
                       alpha=config["alpha"]
                       )
    else:
        train_transformer(config["model"],
                          dataset,
                          config["output_dir"],
                          training_batch_size=config["batch_size"],
                          eval_batch_size=config["batch_size"],
                          learning_rate=config["learning_rate"],
                          num_train_epochs=config["num_train_epochs"],
                          weight_decay=config["weight_decay"])


if __name__ == '__main__':
    main()
