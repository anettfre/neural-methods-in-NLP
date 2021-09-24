from argparse import ArgumentParser
import pandas as pd
import pickle
import zipfile
import torch
from sklearn.metrics import accuracy_score
from section6 import RNNModel, TSVDataset
from gensim.models import KeyedVectors
from torch import nn
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F


if __name__ == "__main__":
    # add command line arguments
    # this is probably the easiest way to store args for downstream
    parser = ArgumentParser()
    parser.add_argument(
        "--test",
        "-t",
        required=True,
        help="path to a file with test data",
        default="stanford_sentiment_binary.tsv.gz",
    )
    args = parser.parse_args()

    with zipfile.ZipFile("/cluster/shared/nlpl/data/vectors/latest/40.zip", "r") as archive:
        stream = archive.open("model.bin")
    word2vec = KeyedVectors.load_word2vec_format(stream, binary = True)
    #word2vec = KeyedVectors.load_word2vec_format("model.bin", binary = True)

    word2vec.add('<unk>', weights = torch.rand(word2vec.vector_size))
    word2vec.add('<pad>', weights = torch.zeros(word2vec.vector_size))

    def pad_batches(batch, pad_X):
        #padding batch
        longest_sentence = max([X.size(0) for X, y in batch])
        new_X = torch.stack([F.pad(X, (0, longest_sentence - X.size(0)), value=pad_X) for X, y in batch])
        new_y = torch.LongTensor([y for X, y in batch])
        lengths = torch.LongTensor([X.size(0) for X, y in batch])
        return (new_X, lengths), new_y

    # load vocab
    with open("/cluster/projects/nn9851k/anthi/vocab.pickle", 'rb') as f:
        vocab = pickle.load(f)

    df = pd.read_csv(args.test, sep='\t', header=0, compression='gzip')
    df = df[df.label != 'label']

    test_dataset = TSVDataset(df, embedder=word2vec, vocab=vocab)
    pad_token = test_dataset.embedder.vocab['<pad>'].index
    test_iter = data.DataLoader(test_dataset, batch_size=len(test_dataset), collate_fn = lambda x: pad_batches(x, pad_token))

    #get model from saga
    saved_model = torch.load("/cluster/projects/nn9851k/anthi/model_state.pth")
    #creating model
    model = RNNModel(embedder = word2vec, num_labels = 2, architecture = "lstm", representation = "max_pool", layers=2, bidir=True) 
    model.load_state_dict(saved_model)
    model.eval()

    #writing to file, in the same format as 
    #stanford_sentiment_binary file
    with open("predictions.tsv", 'w') as f:
        f.write("labels\ttokens\tlemmatized")
        f.write("\n")
        for i, (text, label) in enumerate(test_iter):
            y_pred = model(text)
            for j in range(len(text[0])):
                if y_pred[j].argmax() == 0:
                    f.write("positive")
                else:
                    f.write("negative")
                f.write("\t")
                f.write(df.iloc[j][1])
                f.write("\t")
                f.write(df.iloc[j][2])
   
                f.write("\n")

