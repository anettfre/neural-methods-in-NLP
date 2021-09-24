from torch.utils.data import DataLoader, Dataset
import torch
from collections import Counter


class TSVDataset(Dataset):
    #code is inspired from solution to mandatory1
    def __init__(self, args, data, vocab=None, embeddings_model=None):
        self.tokens = list(data['tokens'].str.split(" "))
        self.label = list(data['label'])
        
        self.embeddings = embeddings_model.wv
        self.dimensions = embeddings_model.vector_size

        if not vocab:
            label_vocab = list(set(self.label))
            self.vocab = {'label': label_vocab}

        else:
            self.vocab = vocab

        self.num_features = args.vocab_size
        self.num_labels = len(self.vocab['label'])

        label_indexer = {i: n for n, i in enumerate(self.vocab['label'])}
        self.indexers = {'label': label_indexer}
        
    def __getitem__(self, index):
        current_tokens = self.tokens[index]
        current_label = self.label[index]
        indices = list()
        for i in current_tokens:
            #if i in self.vocab['tokens']:
                #X[self.indexers['tokens'][i]] += 1
            #if i in self.embeddings:
            #    X += self.embeddings[i]
            if i in self.embeddings:
                indices.append(self.embeddings.vocab[i].index)

        X = torch.LongTensor(indices)
        y = torch.LongTensor([self.indexers['label'][current_label]])
        return X, y
    
    def __len__(self):
        return len(self.tokens)
