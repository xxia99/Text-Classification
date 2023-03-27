# Author: Xin Xia
# Date: 3/26/2023
##########################
#     Util Functions     #
##########################
import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from model import TextMLP


def load_data(tokenizer, dataframe, seed=2, split=0.2):
    """
    :param tokenizer: the loaded tokenizer
    :param dataframe: the original dataframe with text descriptions
    :param split: the ratio of test data split from the original dataframe, default to be 0.2.
    This function first splits the dataframe with text descriptions into training and test set.
    Then, texts are encoded with loaded tokenizer. Last, build dataset with tokenized text.
    """
    train_split, test_split = train_test_split(dataframe, test_size=split, random_state=seed)
    train_texts, train_labels = train_split['text_description'].tolist(), train_split['target_label'].tolist()
    test_texts, test_labels = test_split['text_description'].tolist(), test_split['target_label'].tolist()
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_encodings['input_ids']),
                                                   torch.tensor(train_encodings['attention_mask']),
                                                   torch.tensor(train_labels))
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(test_encodings['input_ids']),
                                                  torch.tensor(test_encodings['attention_mask']),
                                                  torch.tensor(test_labels))
    return train_dataset, test_dataset


def select_embedding_layer(use_pretrained, tokenizer, device):
    if not use_pretrained:
        model = TextMLP(use_pretrained=use_pretrained, vocab_size=tokenizer.vocab_size).to(device)
    else:
        model = TextMLP().to(device)
    return model


def smooth(d, rolling_intv):
    df = pd.DataFrame(d)
    d = list(np.hstack(df.rolling(rolling_intv, min_periods=1).mean().values))
    return d


def tensor_to_list(loss_list):
    loss_list_copy = []
    for i in range(len(loss_list)):
        loss_list_copy.append(loss_list[i].item())
    return loss_list_copy


def plot_loss(loss_list, use_pretrained):
    if use_pretrained:
        preorun = "Pretrained"
    else:
        preorun = "Unpretrained"
    steps = list(range(len(loss_list)))
    loss_list_copy = tensor_to_list(loss_list)
    loss_list_copy_smooth = smooth(loss_list_copy, 100)
    plt.figure()
    plt.plot(steps, loss_list_copy, '-', color='red', alpha=0.2)
    plt.plot(steps, loss_list_copy_smooth, '-', color='red')
    plt.xlabel('steps')
    plt.ylabel('training loss')
    plt.title(f'TextMLP with {preorun} Embedding Layer')
    plt.grid(True, linestyle=':', alpha=0.8)
    plt.show()


def train_model(model, loss_function, optimizer, train_dataset, num_epochs, device, batch_size):
    train_data = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
    loss_list = []
    for epoch in range(num_epochs):
        with tqdm(total=int(len(train_data)), desc="epoch%d" % epoch) as pbar:
            running_loss = 0.0
            running_corrects = 0.0
            model.train()
            for batch in train_data:
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                labels = batch[2].to(device)
                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask)
                _, preds = torch.max(outputs, 1)
                loss = loss_function(outputs, labels)
                loss_list.append(loss)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * input_ids.size(0)
                running_corrects += torch.sum(preds == labels.data)
                pbar.update(1)
            epoch_loss = running_loss / len(train_data.dataset)
            epoch_acc = running_corrects.double() / len(train_data.dataset)
            print('Epoch {}/{} \nTrain Loss: {:.4f} Acc: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss, epoch_acc))
    return model, loss_list


def evaluate(model, loss_function, optimizer, test_dataset, num_epochs, device, batch_size):
    test_data = DataLoader(test_dataset, sampler=RandomSampler(test_dataset), batch_size=batch_size)
    for epoch in range(num_epochs):
        with tqdm(total=int(len(test_data))) as pbar:
            model.eval()
            running_loss = 0.0
            running_corrects = 0.0
            for batch in test_data:
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                labels = batch[2].to(device)

                outputs = model(input_ids, attention_mask)
                _, preds = torch.max(outputs, 1)
                loss = loss_function(outputs, labels)
                running_loss += loss.item() * input_ids.size(0)
                running_corrects += torch.sum(preds == labels.data)
                pbar.update(1)
            epoch_loss = running_loss / len(test_data.dataset)
            epoch_acc = running_corrects.double() / len(test_data.dataset)
            print('Val Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))