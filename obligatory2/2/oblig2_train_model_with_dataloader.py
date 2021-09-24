from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from argparse import ArgumentParser
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import torch
import tqdm
from torch import nn
from torch.utils import data
import pickle
import numpy as np
import gensim
import logging
import zipfile
import json
import random
import time

import torch.nn.functional as F
from loader import TSVDataset
from models import Classifier
from torch.utils.data import DataLoader

def load_embedding(modelfile):
    # Binary word2vec format:
    if modelfile.endswith(".bin.gz") or modelfile.endswith(".bin"):
        emb_model = gensim.models.KeyedVectors.load_word2vec_format(
            modelfile, binary=True, unicode_errors="replace"
        )
    # Text word2vec format:
    elif (
        modelfile.endswith(".txt.gz")
        or modelfile.endswith(".txt")
        or modelfile.endswith(".vec.gz")
        or modelfile.endswith(".vec")
    ):
        emb_model = gensim.models.KeyedVectors.load_word2vec_format(
            modelfile, binary=False, unicode_errors="replace"
        )
    # ZIP archive from the NLPL vector repository:
    elif modelfile.endswith(".zip"):
        with zipfile.ZipFile(modelfile, "r") as archive:
            # Loading and showing the metadata of the model:
            metafile = archive.open("meta.json")
            metadata = json.loads(metafile.read())
            for key in metadata:
                print(key, metadata[key])
            print("============")
            # Loading the model itself:
            stream = archive.open(
                "model.bin"  # or model.txt, if you want to look at the model
            )
            emb_model = gensim.models.KeyedVectors.load_word2vec_format(
                stream, binary=True, unicode_errors="replace"
            )
    else:  # Native Gensim format?
        emb_model = gensim.models.KeyedVectors.load(modelfile)
        #  If you intend to train the model further:
        # emb_model = gensim.models.Word2Vec.load(embeddings_file)
    # Unit-normalizing the vectors (if they aren't already):
    emb_model.init_sims(
        replace=True
    )
    return emb_model

def evaluate(model, data, labels=None):
    #evalutate the model, returns scores
    gold, predictions = [], []
    for n, (input_data, gold_label) in enumerate(data):
        out = model(input_data)
        predicted = out.argmax(axis=1)
        gold.extend(gold_label.tolist())
        predictions.extend(predicted.tolist())

    if labels:
        print(metrics.classification_report(gold, predictions, target_names=labels))

    return metrics.accuracy_score(gold, predictions)
    #return metrics.f1_score(gold, predictions, average='macro')

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--path", default="stanford_sentiment_binary.tsv.gz")
    parser.add_argument("--save", default="test")
    parser.add_argument("--vocab_size", action="store", type=int, default=2000)
    parser.add_argument("--hidden_dim", action="store", type=int, default=128)
    parser.add_argument("--batch_size", action="store", type=int, default=32)
    parser.add_argument("--lr", action="store", type=float, default=1e-3)
    parser.add_argument("--epochs", action="store", type=int, default=15)
    parser.add_argument("--split", action="store", type=float, default=0.8)

    parser.add_argument("--embeddings", action="store", default="40.zip")

    args = parser.parse_args()

    #setting a seed for reproducibility
    torch.manual_seed(42)

    print("Loading the embeddings...\n")

    embeddings_model = load_embedding(args.embeddings)
    args.vocab_size = len(embeddings_model.wv.vocab)

    embeddings_model.add('<pad>', weights=torch.zeros(embeddings_model.vector_size))

    print("Loading the dataset...\n")
    
    df = pd.read_csv(args.path, sep='\t', header=0, compression='gzip')
    df = df[df.label != 'label']

    #split the dataset in train and val with a given split given from args, defalut=0.2
    train_df, val_df = train_test_split(df, train_size=args.split)
    train_dataset = TSVDataset(args, train_df, embeddings_model=embeddings_model)
    val_dataset = TSVDataset(args, val_df, vocab=train_dataset.vocab, embeddings_model=train_dataset.embeddings)

    print(train_dataset.indexers['label'])

    model = Classifier(args, train_dataset.num_labels, embeddings_model)
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)

    val_labels = []
    val_predictions = []

    def pad_batches(batch, pad_idx):
        #padding the batches
        longest_sentence = max([X.size(0) for X, y in batch])

        new_X = torch.stack([F.pad(X, (0, longest_sentence - X.size(0)), value=pad_idx) for X, y in batch])
        new_y = torch.stack([y for X, y in batch])

        return new_X, new_y

    pad_idx = embeddings_model.vocab['<pad>'].index
    batch_size = 16
    loader = DataLoader(dataset=train_dataset, batch_size=batch_size, collate_fn=lambda x: pad_batches(x, pad_idx))

    #training, looping over all the batches
    for epoch in range(args.epochs):
        model.train()
        for i, batch in enumerate(tqdm.tqdm(loader)):
            text = batch[0]
            label = batch[1]
            for n in range(len(text)):
                optimiser.zero_grad()
                y_pred = model(text[n])
                loss = criterion(y_pred, label[n])
                loss.backward()
                optimiser.step()

        #evaluation
        model.eval()
        train_f1 = evaluate(model, train_dataset)
        val_f1 = evaluate(model, val_dataset)

        print(f"epoch: {epoch}\tloss: {loss.item():.3f}\tAccuracy: {train_f1:.3f}")
        print(f"Validation Accuracy: {val_f1:.3f}")

    #save the model as a pickle
    torch.save({'model': model.state_dict(),
                'training_args': args}, f"{args.save}_model.pt")