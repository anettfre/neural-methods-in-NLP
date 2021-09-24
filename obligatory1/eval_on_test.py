import warnings
warnings.filterwarnings('ignore')
import torch
import pickle
import pandas as pd
import numpy as np
import time
import pickle
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from argparse import ArgumentParser
from torch.utils import data
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class Model(nn.Module):
    
    def __init__(self, in_features, h1 = 64, h2 = 64, out_features = 20):

        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.out(x))

        return x

def datasetloader(filename, vectorizer, encoder):
    '''Load the dataset and return a dataloader object for test data'''
    df = pd.read_csv(filename, sep = '\t', header = 0, compression = 'gzip')

    words = df['text']
    classes = df['source']

    in_features = vectorizer.transform(words.values).toarray().astype(np.float32)
    labels = encoder.transform(classes.values)

    X_test = torch.from_numpy(in_features).type(torch.float)
    y_test = torch.from_numpy(labels).type(torch.float)

    test = TensorDataset(X_test, y_test)

    test_batch = DataLoader(test, batch_size=len(test), shuffle=False)

    return test_batch

def evaluate(model, dataloader):
    y_pred = []
    y_true = []
    for x, y in dataloader:
        y_true.append(y)
        y_eval = model.forward(x)
        predictions = y_eval.max(dim=1)[1]
        y_pred.append(predictions)


    print(accuracy_score(y_true[0], y_pred[0]))
    print(classification_report(y_true[0], y_pred[0], digits=4))


if __name__ == "__main__":
    parser = ArgumentParser()
    # we supply the path to the held-out test set
    # NB: make sure the argument is called "--test_path"
    parser.add_argument("--test_path", action='store')

    #Load the Torch model
    parser.add_argument("--model", default="model_state.pth")
    #Vectorizer for transforming input documents X
    parser.add_argument("--vectorizer", default="/cluster/projects/nn9851k/anthi/vectorizer.pickle")

    #Label Encoder for output targets y
    parser.add_argument("--encoder", default="encoder.pickle")
    args = parser.parse_args()

    # it helps to save your model as a dict
    # with hyperparameters also saved within them
    saved_model = torch.load(args.model)
    with open(args.vectorizer, "rb") as file:
        vectorizer = pickle.load(file)
        file.close()
    with open(args.encoder, "rb") as file:
        labelencoder = pickle.load(file)
        file.close()

    # Load the Dataset with this function (not using a custom dataset object)
    test_iter = datasetloader(args.test_path, vectorizer, labelencoder)

    print("Dataset loaded\n")

    # args here should include eg. hidden dim, vocab sizes (both text and labels), etc.
    # note that you have probably already saved the hidden dim :)
    model = Model(in_features=2000)
    model.load_state_dict(saved_model)

    print("Model loaded\n")

    model.eval()
    # Using evaluate as in oblig1code.py
    evaluate(model, test_iter)
