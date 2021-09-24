from argparse import ArgumentParser
import pandas as pd
import pickle
import zipfile
import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from oblig3_basic import ConnluDataset, VecModel, collate_fn, build_mask
import transformers
from conllu import parse, parse_tree
import tqdm


if __name__ == "__main__":
    #Add command line arguments
    parser = ArgumentParser()
    parser.add_argument(
        "--test",
        "-t",
        required=False,
        help="path to a file with test data",
        default="/cluster/projects/nn9851k/IN5550/norne-nb-in5550-train.conllu",
    )

    args = parser.parse_args()
    
    #Get saved model dictionary from saga
    saved_model = torch.load("/cluster/projects/nn9851k/IN5550/walker/basic_model.pt")
    vocab = saved_model['label_vocab']

    #Load model parameters
    norbert = "/cluster/shared/nlpl/data/vectors/latest/216/"
    tokenizer = transformers.BertTokenizer.from_pretrained(norbert)
    model = VecModel(norbert = norbert, num_labels = len(vocab))
    model.load_state_dict(saved_model['model'])
    model.eval()

    data = parse(open(args.test, "r").read())
    test_dataset = ConnluDataset(data, label_vocab=vocab)
    test_iter = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)

    predictions = []
    for text, label in tqdm.tqdm(test_iter):
        X = tokenizer(text, is_split_into_words=True, return_tensors='pt', padding=True)['input_ids']
        batch_mask = build_mask(tokenizer, X)
        y_pred = model(X, batch_mask)
        y_pred = y_pred.permute(0, 2, 1)
        y_pred = y_pred.argmax(dim=1).detach().numpy()
        for i in y_pred:
            predictions.append([test_dataset.inverse_indexer[int(p)] for p in i])


    for i, sentence in enumerate(data):
        sl = len(sentence)
        p = predictions[i][0:sl]
        for j, token in enumerate(sentence):
            token['misc']['name'] = p[j]
     
    with open('predictions.conllu', 'w') as f:
        f.writelines([sentence.serialize() + "\n" for sentence in data])
