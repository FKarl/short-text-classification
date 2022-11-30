import logging

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import get_linear_schedule_with_warmup, AdamW

from data import Dataset, SimpleDataset, prepare_data
from models import collate_for_mlp, collate_for_transformer


class SimplifiedStacking:
    """
    A simple stacking model that uses two models and a meta model to predict the labels.
    """
    def __init__(self, model1, model2, meta_model, is_m1_transformer, is_m2_transformer, is_mm_transformer, tokenizer1,
                 tokenizer2, tokenizer3):
        self.model1 = model1
        self.model2 = model2
        self.meta_model = meta_model
        self.is_m1_transformer = is_m1_transformer
        self.is_m2_transformer = is_m2_transformer
        self.is_mm_transformer = is_mm_transformer
        self.tokenizer1 = tokenizer1
        self.tokenizer2 = tokenizer2
        self.tokenizer3 = tokenizer3
        self.trained = False

    def fit(self, dataset, batch_size, m1_lr, m2_lr, mm_lr, m1_weight_decay,
            m2_weight_decay, mm_weight_decay, epochs, device, m1_num_warmup_steps=0, m2_num_warmup_steps=0,
            mm_num_warmup_steps=0):
        """
        Fit the models to the dataset, by training model1 normal, model2 on the misclassified examples of model1,
        and the meta model decide which model to use.

        :param dataset: Dataset object
        :param batch_size: batch size
        :param m1_lr: learning rate for model 1
        :param m2_lr: learning rate for model 2
        :param mm_lr: learning rate for meta model
        :param m1_weight_decay: weight decay for model 1
        :param m2_weight_decay: weight decay for model 2
        :param mm_weight_decay: weight decay for meta model
        :param epochs: number of epochs
        :param device: device to use
        :param m1_num_warmup_steps: number of warmup steps for model 1
        :param m2_num_warmup_steps: number of warmup steps for model 2
        :param mm_num_warmup_steps: number of warmup steps for meta model
        :return: None
        """
        logging.debug("Starting to fit")
        # prerequisites
        _, train_data, label_dict = prepare_data(dataset, self.tokenizer1,
                                                 Dataset if self.is_m1_transformer else SimpleDataset, shuffle=True)

        train_loader_m1 = DataLoader(train_data,
                                     collate_fn=collate_for_transformer if self.is_m1_transformer else collate_for_mlp,
                                     batch_size=batch_size,
                                     shuffle=False)  # this has to be False, so the indexing is correct
        optimiser_m1 = AdamW(self.model1.parameters(), lr=m1_lr, weight_decay=m1_weight_decay)
        scheduler_m1 = get_linear_schedule_with_warmup(optimiser_m1, num_warmup_steps=m1_num_warmup_steps,
                                                       num_training_steps=len(train_loader_m1) * epochs)

        optimiser_m2 = AdamW(self.model2.parameters(), lr=m2_lr, weight_decay=m2_weight_decay)

        optimiser_mm = AdamW(self.meta_model.parameters(), lr=mm_lr, weight_decay=mm_weight_decay)

        logging.debug("Starting to train")
        # train
        self.model1.train()
        train_iterator = trange(epochs, desc="Epoch")
        for epoch in train_iterator:
            train_iterator.set_description(f"Epoch {epoch}")
            correct_classified_inputs = []
            misclassified_inputs = []
            data_counter = 0

            # region train model 1
            epoch_iterator = tqdm(train_loader_m1, desc="Model 1 Iteration")
            for batch in epoch_iterator:
                batch = tuple(t.to(device) for t in batch)
                if self.is_m1_transformer:
                    inputs = {'input_ids': batch[0],
                              'attention_mask': batch[1],
                              'labels': batch[2]}
                    outputs = self.model1(**inputs)
                else:
                    flat_docs, offsets, labels = batch
                    outputs = self.model1(flat_docs, offsets, labels)
                    inputs = {'input_ids': flat_docs,
                              'labels': labels}

                loss = outputs[0]
                logits = outputs[1]

                # collect misclassified inputs and labels
                for i in range(len(logits)):
                    if not torch.equal(inputs['labels'][i], torch.argmax(logits[i])):
                        misclassified_inputs.append(data_counter)
                    else:
                        correct_classified_inputs.append(data_counter)
                    data_counter += 1

                loss.backward()
                optimiser_m1.step()
                scheduler_m1.step()
                optimiser_m1.zero_grad()
            # endregion

            # region train model 2
            logging.debug(f"Misclassified {len(misclassified_inputs)} inputs")
            logging.debug(f"Correctly classified {len(correct_classified_inputs)} inputs")
            # train model 2 on the missclassified inputs
            if len(misclassified_inputs) > 0:

                # select only the misclassified inputs
                train_text, train_labels = dataset['train']
                train_text = [train_text[i] for i in misclassified_inputs]
                train_labels = [train_labels[i] for i in misclassified_inputs]
                train_encodings = self.tokenizer2(train_text, truncation=True, padding=True)
                train_labels_encoded = [label_dict[label] for label in train_labels]
                m2_dataset = Dataset(train_encodings,
                                     train_labels_encoded) if self.is_m2_transformer else SimpleDataset(
                    train_encodings,
                    train_labels_encoded)

                train_loader_m2 = DataLoader(m2_dataset,
                                             collate_fn=collate_for_transformer if self.is_m2_transformer else collate_for_mlp,
                                             batch_size=batch_size,
                                             shuffle=True)
                scheduler_m2 = get_linear_schedule_with_warmup(optimiser_m2, num_warmup_steps=m2_num_warmup_steps,
                                                               num_training_steps=len(train_loader_m2) * epochs)

                epoch_iterator2 = tqdm(train_loader_m2, desc="Model 2 Iteration")
                for batch2 in epoch_iterator2:
                    batch2 = tuple(t.to(device) for t in batch2)
                    if self.is_m2_transformer:
                        inputs = {'input_ids': batch2[0],
                                  'attention_mask': batch2[1],
                                  'labels': batch2[2]}
                        outputs = self.model2(**inputs)
                    else:
                        flat_docs, offsets, labels = batch2
                        outputs = self.model1(flat_docs, offsets, labels)
                    loss = outputs[0]

                    loss.backward()
                    optimiser_m2.step()
                    optimiser_m2.step()
                    scheduler_m2.step()
                    optimiser_m2.zero_grad()
            # endregion

            # region train meta model to distinguish between correct and misclassified inputs
            train_text, _ = dataset['train']
            correct = [train_text[i] for i in correct_classified_inputs]
            misclassified = [train_text[i] for i in misclassified_inputs]
            X = correct + misclassified
            X = self.tokenizer3(X, truncation=True, padding=True)
            y = [0] * len(correct) + [1] * len(misclassified)
            meta_dataset = Dataset(X, y) if self.is_mm_transformer else SimpleDataset(X, y)
            train_loader_meta = DataLoader(meta_dataset,
                                           collate_fn=collate_for_transformer if self.is_mm_transformer else collate_for_mlp,
                                           batch_size=batch_size,
                                           shuffle=True)
            scheduler_mm = get_linear_schedule_with_warmup(optimiser_mm, num_warmup_steps=mm_num_warmup_steps,
                                                           num_training_steps=len(train_loader_meta) * epochs)
            epoch_iterator3 = tqdm(train_loader_meta, desc="Meta Iteration")
            for batch3 in epoch_iterator3:
                batch3 = tuple(t.to(device) for t in batch3)
                if self.is_mm_transformer:
                    inputs = {'input_ids': batch3[0],
                              'attention_mask': batch3[1],
                              'labels': batch3[2]}
                    outputs = self.meta_model(**inputs)
                else:
                    flat_docs, offsets, labels = batch3
                    outputs = self.meta_model(flat_docs, offsets, labels)
                loss = outputs[0]

                loss.backward()
                optimiser_mm.step()
                scheduler_mm.step()
                optimiser_mm.zero_grad()
            # endregion

            self.trained = True

    def evaluate(self, dataset, batch_size, device):
        assert self.trained, "Model not trained yet"

        m1_data = []
        m2_data = []

        # region meta model decides which model to use
        test_data, _, label_dict = prepare_data(dataset, self.tokenizer3,
                                                Dataset if self.is_mm_transformer else SimpleDataset, shuffle=True)

        self.meta_model.to(device)
        data_counter = 0
        data_loader_mm = DataLoader(test_data,
                                    collate_fn=collate_for_transformer if self.is_mm_transformer else collate_for_mlp,
                                    batch_size=batch_size,
                                    shuffle=False)  # this has to be False, so the indexing is correct
        for batch in tqdm(data_loader_mm, desc="Meta Model"):
            self.meta_model.eval()
            batch = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                if self.is_mm_transformer:
                    inputs = {'input_ids': batch[0],
                              'attention_mask': batch[1],
                              'labels': batch[2]}
                    outputs = self.meta_model(**inputs)
                else:
                    flat_docs, offsets, labels = batch
                    outputs = self.meta_model(flat_docs, offsets, labels)

            loss, logits = outputs[:2]
            for logit in logits:
                if logit[0] > logit[1]:
                    m1_data.append(data_counter)
                else:
                    m2_data.append(data_counter)
                data_counter += 1
        # endregion

        logging.info(
            f"Meta model decided to use model 1 for {len(m1_data)} inputs and model 2 for {len(m2_data)} inputs")

        # region evaluate model 1
        preds_m1 = []
        labels_m1 = []
        if len(m1_data) > 0:
            test_data, test_labels = dataset['test']
            label_dict = dataset['label_dict']
            selected_test_data = [test_data[i] for i in m1_data]
            selected_test_labels = [label_dict[test_labels[i]] for i in m1_data]
            test_encodings = self.tokenizer1(selected_test_data, truncation=True, padding=True)
            test_dataset = Dataset(test_encodings, selected_test_labels) if self.is_m1_transformer else SimpleDataset(
                test_encodings,
                selected_test_labels)
            test_loader = DataLoader(test_dataset,
                                     collate_fn=collate_for_transformer if self.is_m1_transformer else collate_for_mlp,
                                     batch_size=batch_size,
                                     shuffle=True)
            self.model1.to(device)
            self.model1.eval()
            for batch in tqdm(test_loader, desc="Evaluate Model 1"):
                batch = tuple(t.to(device) for t in batch)
                with torch.no_grad():
                    if self.is_m1_transformer:
                        inputs = {'input_ids': batch[0],
                                  'attention_mask': batch[1],
                                  'labels': batch[2]}
                        labels = inputs['labels']
                        outputs = self.model1(**inputs)
                    else:
                        flat_docs, offsets, labels = batch
                        outputs = self.model1(flat_docs, offsets, labels)

                loss, logits = outputs[:2]
                preds_m1.extend(logits.detach().cpu().numpy())
                labels_m1.extend(labels.detach().cpu().numpy())

            preds_m1 = np.argmax(preds_m1, axis=1)
        # endregion

        # region evaluate model 2
        preds_m2 = []
        labels_m2 = []
        if len(m2_data) > 0:
            test_data, test_labels = dataset['test']
            label_dict = dataset['label_dict']
            selected_test_data = [test_data[i] for i in m2_data]
            selected_test_labels = [label_dict[test_labels[i]] for i in m2_data]
            test_encodings = self.tokenizer2(selected_test_data, truncation=True, padding=True)
            test_dataset = Dataset(test_encodings, selected_test_labels) if self.is_m2_transformer else SimpleDataset(
                test_encodings,
                selected_test_labels)
            test_loader = DataLoader(test_dataset,
                                     collate_fn=collate_for_transformer if self.is_m2_transformer else collate_for_mlp,
                                     batch_size=batch_size,
                                     shuffle=True)
            self.model2.to(device)
            self.model2.eval()

            for batch in tqdm(test_loader, desc="Evaluate Model 2"):
                batch = tuple(t.to(device) for t in batch)
                with torch.no_grad():
                    if self.is_m2_transformer:
                        inputs = {'input_ids': batch[0],
                                  'attention_mask': batch[1],
                                  'labels': batch[2]}
                        labels = inputs['labels']
                        outputs = self.model2(**inputs)
                    else:
                        flat_docs, offsets, labels = batch
                        outputs = self.model2(flat_docs, offsets, labels)

                loss, logits = outputs[:2]
                preds_m2.extend(logits.detach().cpu().numpy())
                labels_m2.extend(labels.detach().cpu().numpy())

            preds_m2 = np.argmax(preds_m2, axis=1)
        # endregion

        # calculate accuracy
        preds = np.concatenate((preds_m1, preds_m2))
        labels = np.concatenate((labels_m1, labels_m2))
        acc = accuracy_score(labels, preds)

        logging.info(f"Accuracy: {acc}")

        return acc


class SimplifiedWeightedBoost:
    """
    A simple ensemble model that trains the second model on the misclassified examples of the first model.
    """
    def __init__(self, model1, model2, is_m1_transformer, is_m2_transformer, tokenizer1,
                 tokenizer2):
        self.model1 = model1
        self.model2 = model2
        self.is_m1_transformer = is_m1_transformer
        self.is_m2_transformer = is_m2_transformer
        self.tokenizer1 = tokenizer1
        self.tokenizer2 = tokenizer2
        self.trained = False

    def fit(self, dataset, batch_size, m1_lr, m2_lr, m1_weight_decay,
            m2_weight_decay, epochs, device, m1_num_warmup_steps=0, m2_num_warmup_steps=0):
        """
        Fit the models to the dataset, by training model1 normal and model2 on the misclassified examples of model1.

        :param dataset: Dataset object
        :param batch_size: batch size
        :param m1_lr: learning rate for model 1
        :param m2_lr: learning rate for model 2
        :param m1_weight_decay: weight decay for model 1
        :param m2_weight_decay: weight decay for model 2
        :param epochs: number of epochs
        :param device: device to use
        :param m1_num_warmup_steps: number of warmup steps for model 1
        :param m2_num_warmup_steps: number of warmup steps for model 2
        :return: None
        """
        logging.debug("Starting to fit")
        # prerequisites
        _, train_data, label_dict = prepare_data(dataset, self.tokenizer1,
                                                 Dataset if self.is_m1_transformer else SimpleDataset, shuffle=True)

        train_loader_m1 = DataLoader(train_data,
                                     collate_fn=collate_for_transformer if self.is_m1_transformer else collate_for_mlp,
                                     batch_size=batch_size,
                                     shuffle=False)  # this has to be False, so the indexing is correct
        optimiser_m1 = AdamW(self.model1.parameters(), lr=m1_lr, weight_decay=m1_weight_decay)
        scheduler_m1 = get_linear_schedule_with_warmup(optimiser_m1, num_warmup_steps=m1_num_warmup_steps,
                                                       num_training_steps=len(train_loader_m1) * epochs)

        optimiser_m2 = AdamW(self.model2.parameters(), lr=m2_lr, weight_decay=m2_weight_decay)

        logging.debug("Starting to train")
        # train
        self.model1.train()
        train_iterator = trange(epochs, desc="Epoch")
        for epoch in train_iterator:
            train_iterator.set_description(f"Epoch {epoch}")
            correct_classified_inputs = []
            misclassified_inputs = []
            data_counter = 0

            # region train model 1
            epoch_iterator = tqdm(train_loader_m1, desc="Model 1 Iteration")
            for batch in epoch_iterator:
                batch = tuple(t.to(device) for t in batch)
                if self.is_m1_transformer:
                    inputs = {'input_ids': batch[0],
                              'attention_mask': batch[1],
                              'labels': batch[2]}
                    outputs = self.model1(**inputs)
                else:
                    flat_docs, offsets, labels = batch
                    outputs = self.model1(flat_docs, offsets, labels)
                    inputs = {'input_ids': flat_docs,
                              'labels': labels}

                loss = outputs[0]
                logits = outputs[1]

                # collect misclassified inputs and labels
                for i in range(len(logits)):
                    if not torch.equal(inputs['labels'][i], torch.argmax(logits[i])):
                        misclassified_inputs.append(data_counter)
                    else:
                        correct_classified_inputs.append(data_counter)
                    data_counter += 1

                loss.backward()
                optimiser_m1.step()
                scheduler_m1.step()
                optimiser_m1.zero_grad()
            # endregion

            # region train model 2
            logging.debug(f"Misclassified {len(misclassified_inputs)} inputs")
            logging.debug(f"Correctly classified {len(correct_classified_inputs)} inputs")
            # train model 2 on the missclassified inputs
            if len(misclassified_inputs) > 0:

                # select only the misclassified inputs
                train_text, train_labels = dataset['train']
                train_text = [train_text[i] for i in misclassified_inputs]
                train_labels = [train_labels[i] for i in misclassified_inputs]
                train_encodings = self.tokenizer2(train_text, truncation=True, padding=True)
                train_labels_encoded = [label_dict[label] for label in train_labels]
                m2_dataset = Dataset(train_encodings,
                                     train_labels_encoded) if self.is_m2_transformer else SimpleDataset(
                    train_encodings,
                    train_labels_encoded)

                train_loader_m2 = DataLoader(m2_dataset,
                                             collate_fn=collate_for_transformer if self.is_m2_transformer else collate_for_mlp,
                                             batch_size=batch_size,
                                             shuffle=True)
                scheduler_m2 = get_linear_schedule_with_warmup(optimiser_m2, num_warmup_steps=m2_num_warmup_steps,
                                                               num_training_steps=len(train_loader_m2) * epochs)

                epoch_iterator2 = tqdm(train_loader_m2, desc="Model 2 Iteration")
                for batch2 in epoch_iterator2:
                    batch2 = tuple(t.to(device) for t in batch2)
                    if self.is_m2_transformer:
                        inputs = {'input_ids': batch2[0],
                                  'attention_mask': batch2[1],
                                  'labels': batch2[2]}
                        outputs = self.model2(**inputs)
                    else:
                        flat_docs, offsets, labels = batch2
                        outputs = self.model1(flat_docs, offsets, labels)
                    loss = outputs[0]

                    loss.backward()
                    optimiser_m2.step()
                    optimiser_m2.step()
                    scheduler_m2.step()
                    optimiser_m2.zero_grad()
            # endregion
        self.trained = True

    def evaluate(self, dataset, batch_size, device, alpha=0.5):
        assert self.trained, "Model not trained yet"

        # region evaluate model 1
        test_data, _, label_dict = prepare_data(dataset, self.tokenizer1,
                                                Dataset if self.is_m1_transformer else SimpleDataset,
                                                shuffle=False)
        self.model1.eval()
        self.model1.to(device)
        test_loader = DataLoader(test_data,
                                 collate_fn=collate_for_transformer if self.is_m1_transformer else collate_for_mlp,
                                 batch_size=batch_size,
                                 shuffle=False)  # this has to be False, so the indexing is correct

        preds_m1 = []
        self.model1.to(device)
        self.model1.eval()
        for batch in tqdm(test_loader, desc="Evaluate Model 1"):
            batch = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                if self.is_m1_transformer:
                    inputs = {'input_ids': batch[0],
                              'attention_mask': batch[1],
                              'labels': batch[2]}
                    outputs = self.model1(**inputs)
                else:
                    flat_docs, offsets, labels = batch
                    outputs = self.model1(flat_docs, offsets, labels)

            loss, logits = outputs[:2]
            preds_m1.extend(logits.detach().cpu().numpy())
        # endregion

        # region evaluate model 2
        test_data, _, label_dict = prepare_data(dataset, self.tokenizer2,
                                                Dataset if self.is_m2_transformer else SimpleDataset, shuffle=False)
        self.model2.eval()
        self.model2.to(device)
        test_loader = DataLoader(test_data,
                                 collate_fn=collate_for_transformer if self.is_m2_transformer else collate_for_mlp,
                                 batch_size=batch_size,
                                 shuffle=False)  # this has to be False, so the indexing is correct
        preds_m2 = []
        for batch in tqdm(test_loader, desc="Evaluate Model 2"):
            batch = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                if self.is_m2_transformer:
                    inputs = {'input_ids': batch[0],
                              'attention_mask': batch[1],
                              'labels': batch[2]}
                    outputs = self.model2(**inputs)
                else:
                    flat_docs, offsets, labels = batch
                    outputs = self.model2(flat_docs, offsets, labels)

            loss, logits = outputs[:2]
            preds_m2.extend(logits.detach().cpu().numpy())
        # endregion

        # region combine predictions with alpha
        _, test_labels = dataset['test']
        test_labels_encoded = [label_dict[label] for label in test_labels]

        preds_m1 = np.array(preds_m1)
        preds_m2 = np.array(preds_m2)
        preds = alpha * preds_m1 + (1 - alpha) * preds_m2
        preds = np.argmax(preds, axis=1)
        # endregion

        # calculate accuracy
        acc = accuracy_score(test_labels_encoded, preds)

        logging.info(f"Accuracy: {acc}")

        return acc


class WeightedEnsemble:
    """
    A simple ensemble model that combines the predictions of two models with a given weight.
    """
    def __init__(self, model1, model2, is_m1_transformer, is_m2_transformer, tokenizer1,
                 tokenizer2):
        self.model1 = model1
        self.model2 = model2
        self.is_m1_transformer = is_m1_transformer
        self.is_m2_transformer = is_m2_transformer
        self.tokenizer1 = tokenizer1
        self.tokenizer2 = tokenizer2
        self.trained = False

    def fit(self, dataset, batch_size, m1_lr, m2_lr, m1_weight_decay,
            m2_weight_decay, m1_epochs, m2_epochs, device, m1_num_warmup_steps=0, m2_num_warmup_steps=0):
        """
        Fit the models to the dataset, by training each model separately and then combining the predictions

        :param dataset: Dataset object
        :param batch_size: batch size
        :param m1_lr: learning rate for model 1
        :param m2_lr: learning rate for model 2
        :param m1_weight_decay: weight decay for model 1
        :param m2_weight_decay: weight decay for model 2
        :param m1_epochs: number of epochs for model 1
        :param m2_epochs: number of epochs for model 2
        :param device: device to use
        :param m1_num_warmup_steps: number of warmup steps for model 1
        :param m2_num_warmup_steps: number of warmup steps for model 2
        :return: None
        """
        logging.debug("Starting to fit")
        # prerequisites
        _, train_data, label_dict = prepare_data(dataset, self.tokenizer1,
                                                 Dataset if self.is_m1_transformer else SimpleDataset)

        train_loader_m1 = DataLoader(train_data,
                                     collate_fn=collate_for_transformer if self.is_m1_transformer else collate_for_mlp,
                                     batch_size=batch_size,
                                     shuffle=True)
        optimiser_m1 = AdamW(self.model1.parameters(), lr=m1_lr, weight_decay=m1_weight_decay)
        scheduler_m1 = get_linear_schedule_with_warmup(optimiser_m1, num_warmup_steps=m1_num_warmup_steps,
                                                       num_training_steps=len(train_loader_m1) * m1_epochs)

        _, train_data, label_dict = prepare_data(dataset, self.tokenizer2,
                                                 Dataset if self.is_m2_transformer else SimpleDataset)
        train_loader_m2 = DataLoader(train_data,
                                     collate_fn=collate_for_transformer if self.is_m2_transformer else collate_for_mlp,
                                     batch_size=batch_size,
                                     shuffle=True)

        optimiser_m2 = AdamW(self.model2.parameters(), lr=m2_lr, weight_decay=m2_weight_decay)
        scheduler_m2 = get_linear_schedule_with_warmup(optimiser_m2, num_warmup_steps=m2_num_warmup_steps,
                                                       num_training_steps=len(train_loader_m1) * m2_epochs)

        logging.debug("Starting to train")
        # region train model 1
        self.model1.train()
        train_iterator = trange(m1_epochs, desc="Model 1 Epoch")
        for epoch in train_iterator:
            train_iterator.set_description(f"Model 1 Epoch {epoch}")
            epoch_iterator = tqdm(train_loader_m1, desc="Model 1 Iteration")
            for batch in epoch_iterator:
                batch = tuple(t.to(device) for t in batch)
                if self.is_m1_transformer:
                    inputs = {'input_ids': batch[0],
                              'attention_mask': batch[1],
                              'labels': batch[2]}
                    outputs = self.model1(**inputs)
                else:
                    flat_docs, offsets, labels = batch
                    outputs = self.model1(flat_docs, offsets, labels)

                loss = outputs[0]

                loss.backward()
                optimiser_m1.step()
                scheduler_m1.step()
                optimiser_m1.zero_grad()
        # endregion

        logging.info("Finished training model 1")
        # region train model 2
        self.model2.train()
        train_iterator = trange(m2_epochs, desc="Model 2 Epoch")
        for epoch in train_iterator:
            train_iterator.set_description(f"Model 2 Epoch {epoch}")
            epoch_iterator = tqdm(train_loader_m2, desc="Model 2 Iteration")
            for batch in epoch_iterator:
                batch = tuple(t.to(device) for t in batch)
                if self.is_m2_transformer:
                    inputs = {'input_ids': batch[0],
                              'attention_mask': batch[1],
                              'labels': batch[2]}
                    outputs = self.model2(**inputs)
                else:
                    flat_docs, offsets, labels = batch
                    outputs = self.model2(flat_docs, offsets, labels)

                loss = outputs[0]

                loss.backward()
                optimiser_m2.step()
                scheduler_m2.step()
                optimiser_m2.zero_grad()
        # endregion
        logging.info("Finished training model 2")

        self.trained = True

    def evaluate(self, dataset, batch_size, device, alpha=0.5):
        assert self.trained, "Model not trained yet"

        # region evaluate model 1
        test_data, _, label_dict = prepare_data(dataset, self.tokenizer1,
                                                Dataset if self.is_m1_transformer else SimpleDataset,
                                                shuffle=False)
        self.model1.eval()
        self.model1.to(device)
        test_loader = DataLoader(test_data,
                                 collate_fn=collate_for_transformer if self.is_m1_transformer else collate_for_mlp,
                                 batch_size=batch_size,
                                 shuffle=False)  # this has to be False, so the indexing is correct

        preds_m1 = []
        self.model1.to(device)
        self.model1.eval()
        for batch in tqdm(test_loader, desc="Evaluate Model 1"):
            batch = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                if self.is_m1_transformer:
                    inputs = {'input_ids': batch[0],
                              'attention_mask': batch[1],
                              'labels': batch[2]}
                    outputs = self.model1(**inputs)
                else:
                    flat_docs, offsets, labels = batch
                    outputs = self.model1(flat_docs, offsets, labels)

            loss, logits = outputs[:2]
            preds_m1.extend(logits.detach().cpu().numpy())
        # endregion

        # region evaluate model 2
        test_data, _, label_dict = prepare_data(dataset, self.tokenizer2,
                                                Dataset if self.is_m2_transformer else SimpleDataset, shuffle=False)
        self.model2.eval()
        self.model2.to(device)
        test_loader = DataLoader(test_data,
                                 collate_fn=collate_for_transformer if self.is_m2_transformer else collate_for_mlp,
                                 batch_size=batch_size,
                                 shuffle=False)  # this has to be False, so the indexing is correct
        preds_m2 = []
        for batch in tqdm(test_loader, desc="Evaluate Model 2"):
            batch = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                if self.is_m2_transformer:
                    inputs = {'input_ids': batch[0],
                              'attention_mask': batch[1],
                              'labels': batch[2]}
                    outputs = self.model2(**inputs)
                else:
                    flat_docs, offsets, labels = batch
                    outputs = self.model2(flat_docs, offsets, labels)

            loss, logits = outputs[:2]
            preds_m2.extend(logits.detach().cpu().numpy())
        # endregion

        # region combine predictions with alpha
        _, test_labels = dataset['test']
        test_labels_encoded = [label_dict[label] for label in test_labels]

        preds_m1 = np.array(preds_m1)
        preds_m2 = np.array(preds_m2)
        preds = alpha * preds_m1 + (1 - alpha) * preds_m2
        preds = np.argmax(preds, axis=1)
        # endregion

        # calculate accuracy
        acc = accuracy_score(test_labels_encoded, preds)

        logging.info(f"Accuracy: {acc}")

        return acc
