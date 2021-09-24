import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from collections import Counter
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from argparse import ArgumentParser
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import tqdm
from torch import nn
from torch.utils import data
import pickle
import numpy as np
import gensim
import logging
import json
import random
import time
import zipfile

class RNNModel(nn.Module):
    #the model, argumetns for which model architecture to use and other parameters 
    def __init__(self, embedder, num_labels, architecture = None, representation = None, layers=2, bidir=True):
        super().__init__()
        self._embed = nn.Embedding.from_pretrained(torch.FloatTensor(embedder.vectors))

        self.architecture = architecture
        self.representation = representation

        if self.architecture == "rnn":
            self._rnn = nn.RNN(100, 200, num_layers=layers, dropout = 0.2, batch_first=True, bidirectional=bidir)
        elif self.architecture == "lstm":
            self._rnn = nn.LSTM(100, 200, num_layers=layers, dropout = 0.2, batch_first=True, bidirectional=bidir)
        elif self.architecture == "gru":
            self._rnn = nn.GRU(100, 200, num_layers=layers, dropout = 0.2, batch_first=True, bidirectional=bidir)


        if bidir==True:
            self._linear  = nn.Linear(400, num_labels)
        else:
            self._linear  = nn.Linear(200, num_labels)
        self.dropout = nn.Dropout(p=0.1)


    def forward(self, X):
        X, lengths = X
        embeds = self._embed(X)
        embeds = self.dropout(embeds)
        embeds = nn.utils.rnn.pack_padded_sequence(embeds, lengths, batch_first=True, enforce_sorted=False)
        states, _ = self._rnn(embeds)
        states = nn.utils.rnn.pad_packed_sequence(states, batch_first=True)[0]
        
        row_indices = torch.arange(0, X.size(0)).long()
        col_indices = lengths - 1

        #choosing which sequential order, max_pool gave the best result
        if self.representation == "last":
            last_tensor = states[row_indices, col_indices, :] 
        elif self.representation == "max_pool":
            last_tensor = states[row_indices, :, :]
            last_tensor, _ = torch.max(last_tensor, 1)
        elif self.representation == "sum":
            last_tensor = states[row_indices, :, :]
            last_tensor = torch.sum(last_tensor, dim=1)

        out = self._linear(last_tensor)
        
        return out


class TSVDataset(Dataset):
    def __init__(self, data, embedder, vocab=None):

        self.tokens = list(data['tokens'].str.split(" "))
        self.label = list(data['label'])

        self.label_vocab = list(set(self.label))
        if vocab:
            self.vocab = vocab
        else:
            self.vocab = self.label_vocab
            self.vocab.extend(['@UNK'])

        self.vocab_indexer = {i: n for n, i in enumerate(self.vocab)}
        self.embedder = embedder

        self.label_indexer = {i: n for n, i in enumerate(self.label_vocab)}

    def __getitem__(self, index):
        current_tokens = self.tokens[index]
        current_label = self.label[index]

        X = torch.LongTensor([self.embedder.vocab[token].index if token in self.embedder.vocab else self.embedder.vocab['<unk>'].index
            for token in current_tokens])

        y = self.label_indexer[current_label]
        y = torch.LongTensor([y])


        return X, y

    def __len__(self):
        return len(self.tokens)


def evaluate(y_pred, y_gold):
    #method for evaluation
    y_pred = y_pred.argmax(dim=-1)
    correct = (y_pred == y_gold).nonzero(as_tuple=False).size(0)
    total = (y_gold != -1).nonzero(as_tuple=False).size(0)
    return correct / total


def pad_batches(batch, pad_X):
    #padding batch
    longest_sentence = max([X.size(0) for X, y in batch])
    new_X = torch.stack([F.pad(X, (0, longest_sentence - X.size(0)), value=pad_X) for X, y in batch])
    new_y = torch.LongTensor([y for X, y in batch])
    lengths = torch.LongTensor([X.size(0) for X, y in batch])
    return (new_X, lengths), new_y


def main(learning_rate=0.001, bidir=True, num_layers=2):
    #training a model with given parameters, using early stopping
    model = RNNModel(embedder = word2vec, num_labels = 2, architecture = "lstm", representation = "max_pool", layers=num_layers, bidir=bidir)

    #finding the total number of parameters in the model for section 6.1
    pytorch_total_params = sum(p.numel() for p in model.parameters())

    '''
    architecture options: 1. rnn
                          2. lstm
                          3. gru
    representation options: 1. last
                            2. sum
                            3. max_pool
    '''

    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)


    max_stagnation = 6
    best_val_loss, best_val_epoch = None, None
    early_stop = False

    #training
    start = time.time()
    for epoch in range(200):
        train_iter = tqdm.tqdm(train)
        model.train()
        for text, label in train_iter:
            optimiser.zero_grad()
            y_pred = model(text)
            loss = criterion(y_pred, label)
            loss.backward()
            optimiser.step() 
      
        model.eval()
        #evaluation
        train_acc = evaluate(y_pred, label)
        print(f"epoch: {epoch}\tloss: {loss.item():.3f}\tAccuracy: {train_acc:.3f}")

        for i, (text, label) in enumerate(val):
            y_val = model(text)
            loss_val = criterion(y_val, label) 
            val_loss = loss_val.item()
            optimiser.zero_grad()

        val_acc = evaluate(y_val, label)
        print(f"Validation Loss: {val_loss:.3f}\tAccuracy: {val_acc:.3f}")
        if best_val_loss is None or best_val_loss < val_loss:
            best_val_loss, best_val_epoch = val_loss, epoch
        if best_val_epoch < epoch - max_stagnation:
            early_stop = True 
            print('Early stop.')
            print(f"Learning rate: {learning_rate} Bidirectional: {bidir} Number of layers {num_layers}")
            end = time.time()
            print(f"Time passed: {(end-start)//60}")
            break
        
if __name__ == "__main__":
    torch.manual_seed(1234)

    #word2vec = KeyedVectors.load_word2vec_format("40model.bin", binary = True)
    #open a model on saga
    with zipfile.ZipFile("/cluster/shared/nlpl/data/vectors/latest/40.zip", "r") as archive:
        stream = archive.open("model.bin")
    word2vec = KeyedVectors.load_word2vec_format(stream, binary = True)


    word2vec.add('<unk>', weights = torch.rand(word2vec.vector_size))
    word2vec.add('<pad>', weights = torch.zeros(word2vec.vector_size))

    df = pd.read_csv("stanford_sentiment_binary.tsv.gz", sep='\t', header=0, compression='gzip')
    df = df[df.label != 'label']

    labels = df['label'].to_list()
    #split in test and validation
    train_df, val_df = train_test_split(df, train_size=0.8)
    train_dataset = TSVDataset(train_df, embedder=word2vec)
    val_dataset = TSVDataset(val_df, embedder=word2vec, vocab = train_dataset.vocab)

    pad_token = train_dataset.embedder.vocab['<pad>'].index

    train = DataLoader(dataset = train_dataset, batch_size=32, shuffle=True, collate_fn = lambda x: pad_batches(x, pad_token))
    val = DataLoader(dataset = val_dataset, batch_size=len(val_dataset), shuffle=False, collate_fn = lambda x: pad_batches(x, pad_token))

    #selected hyperparameters to test in grid search
    lr = [0.0001, 0.001, 0.01]
    bidir = [True, False]
    num_layers = [1,2,3]

    #looping to do grid search
    for l in lr:
        for dir in bidir:
            for lay in num_layers:
                main(l, dir,  lay)


