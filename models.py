import tokenizers
import torch
import torch.nn as nn
import torch.nn.functional as F


def collate_for_mlp(list_of_samples):
    """
    Collate function that creates batches of flat docs tensor and offsets
    Author: Lukas Galke
    https://github.com/lgalke/text-clf-baselines/blob/main/models.py
    """
    offset = 0
    flat_docs, offsets, labels = [], [], []
    for doc, label in list_of_samples:
        if isinstance(doc, tokenizers.Encoding):
            doc = doc.ids
        offsets.append(offset)
        flat_docs.extend(doc)
        labels.append(label)
        offset += len(doc)
    return torch.tensor(flat_docs), torch.tensor(offsets), torch.tensor(labels)


def collate_for_lstm(list_of_samples):
    return collate_for_mlp(list_of_samples)


def collate_for_transformer(list_of_samples):
    """
    Collate function that creates batches of
    returns:
        - input_ids: tensor of shape (batch_size, max_seq_len)
        - attention_mask: tensor
        - labels: tensor of shape (batch_size)
    """
    docs, attention_masks, labels = [], [], []
    for sample in list_of_samples:
        docs.append(sample['input_ids'])
        attention_masks.append(sample['attention_mask'])
        labels.append(int(sample['labels']))
    # to tensors
    docs = torch.stack(docs)
    attention_masks = torch.stack(attention_masks)
    labels = torch.tensor(labels)
    return docs, attention_masks, labels


class MLP(nn.Module):
    """
    Simple MLP
    Author: Lukas Galke
    https://github.com/lgalke/text-clf-baselines/blob/main/models.py
    """

    def __init__(self, vocab_size, num_classes,
                 num_hidden_layers=1,
                 hidden_size=1024, hidden_act='relu',
                 dropout=0.5, idf=None, mode='mean',
                 pretrained_embedding=None, freeze=True,
                 embedding_dropout=0.5):
        nn.Module.__init__(self)
        # Treat TF-IDF mode appropriately
        mode = 'sum' if idf is not None else mode
        self.idf = idf

        # Input-to-hidden (efficient via embedding bag)
        if pretrained_embedding is not None:
            # vocabsize is defined by embedding in this case
            self.embed = nn.EmbeddingBag.from_pretrained(pretrained_embedding, freeze=freeze, mode=mode)
            embedding_size = pretrained_embedding.size(1)
            self.embedding_is_pretrained = True
        else:
            assert vocab_size is not None
            self.embed = nn.EmbeddingBag(vocab_size, hidden_size, mode=mode)
            embedding_size = hidden_size
            self.embedding_is_pretrained = False

        self.activation = getattr(F, hidden_act)
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()

        # Hidden-to-hidden
        for i in range(num_hidden_layers - 1):
            if i == 0:
                self.layers.append(nn.Linear(embedding_size, hidden_size))
            else:
                self.layers.append(nn.Linear(hidden_size, hidden_size))

        # Hidden-to-output
        self.layers.append(nn.Linear(hidden_size if self.layers else embedding_size, num_classes))

        # Loss function
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, input, offsets, labels=None):
        # Use idf weights if present
        idf_weights = self.idf[input] if self.idf is not None else None

        h = self.embed(input, offsets, per_sample_weights=idf_weights)

        if self.idf is not None:
            # In the TF-IDF case: renormalize according to l2 norm
            h = h / torch.linalg.norm(h, dim=1, keepdim=True)

        if not self.embedding_is_pretrained:
            # No nonlinearity when embedding is pretrained
            h = self.activation(h)

        h = self.embedding_dropout(h)

        for i, layer in enumerate(self.layers):
            # at least one
            h = layer(h)
            if i != len(self.layers) - 1:
                # No activation/dropout for final layer
                h = self.activation(h)
                h = self.dropout(h)

        if labels is not None:
            loss = self.loss_function(h, labels)
            return loss, h
        return h


class LSTM(nn.Module):
    """
    Simple LSTM for text classification
    """

    def __init__(self, vocab_size, num_classes, bidirectional, hidden_size, num_layers,
                 dropout, pretrained_embedding=None, freeze=True):
        super(LSTM, self).__init__()

        self.input_size = vocab_size
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        if pretrained_embedding is not None:
            self.embed = nn.EmbeddingBag.from_pretrained(pretrained_embedding, freeze=freeze)
            embedding_dim = pretrained_embedding.size(1)
        else:
            self.embed = nn.EmbeddingBag(vocab_size, hidden_size)
            embedding_dim = hidden_size

        self.embedding = nn.EmbeddingBag(vocab_size, embedding_dim)

        # LSTM architecture
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, dropout=dropout, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)

        # linear layer on top
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), num_classes)

        # Loss function
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, text, offset, label):
        # embedding
        embedded = self.embedding(text, offset)
        embedded = self.dropout(embedded)

        # lstm
        lstm_out, _ = self.lstm(embedded)
        lstm_out = self.dropout(lstm_out)

        # linear layer
        out = self.fc(lstm_out)

        # loss
        loss = self.loss_function(out, label)
        return loss, out
