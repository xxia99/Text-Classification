# Author: Xin Xia
# Date: 3/26/2023
from model import TextMLP
from text_classification import *
import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from model import TextMLP
import os


os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":
    # Load the preprocessed text descriptions data
    df_text = pd.read_csv("preprocessed_data.csv")

    # Hyper parameters for training and test
    batch_size = 4
    learning_rate = 1e-6
    num_epochs = 50
    loss_function = torch.nn.CrossEntropyLoss()  # loss function used in model training

    # Specify the device(cuda or cpu) and tokenizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # load the pretrained tokenizer of BERT

    # Load the training and test data
    seed = 4
    train_dataset, test_dataset = load_data(tokenizer, df_text, seed=seed, split=0.2)

    ##################################
    #  Unpretrained Embedding Layer  #
    ##################################
    use_pretrained = False  # the flag that controls if the model uses an pretrained embedding layer from BERT or not
    model = select_embedding_layer(use_pretrained, tokenizer, device) # Create the model instance from TextMLP() object
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)  # optimizer used in model training
    print("\n####### Training a Model with Unpretrained Embedding Layer ######\n")

    # model training
    trained_model, loss_list_un = train_model(model, loss_function, optimizer, train_dataset, num_epochs, device, batch_size)
    # evaluation
    evaluate(trained_model, loss_function, optimizer, test_dataset, 1, device, batch_size)
    # training loss
    plot_loss(loss_list_un, use_pretrained)

