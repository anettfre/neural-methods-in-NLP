import torch
import transformers
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
from transformers import AdamW
import tqdm
from conllu import parse
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import warnings
from collections import Counter
import math
#import matplotlib.pyplot as plt
#import seaborn as sns
warnings.filterwarnings("ignore")

torch.manual_seed(1234)

class ConnluDataset(Dataset):
    def __init__(self, sentences, label_vocab=None):
        self.text = []
        for sentence in sentences:
            self.text.append([token['form'] for token in sentence])

        self.label = []
        for sentence in sentences:
            self.label.append([token['misc']['name'] for token in sentence])

        self.flat = [item for sublist in self.label for item in sublist]
        if label_vocab == None:
            self.label_vocab = list(set([item for sublist in self.label for item in sublist]))
        else:
            self.label_vocab = label_vocab

        self.label_indexer = {i: n for n, i in enumerate(self.label_vocab)}
        self.inverse_indexer = {n: i for n, i in enumerate(self.label_vocab)}
        
                
    def __getitem__(self, index):
        current_text = self.text[index]
        current_label = self.label[index]
        
        X = current_text
        y = torch.LongTensor([self.label_indexer[i] for i in current_label])
        return X, y
    
    def __len__(self):
        return len(self.text)

class VecModel(nn.Module):
    def __init__(self, norbert, num_labels):
        super().__init__()
        self._bert = BertModel.from_pretrained(norbert)

        for param in self._bert.parameters():
           param.requires_grad = False

        ## selective fine-tuning
        # for name, child in self._bert.named_children():
        #     if name =='embeddings':
        #         for param in child.parameters():
        #             param.requires_grad = True
        #     else:
        #         for param in child.parameters():
        #             param.requires_grad = False

        ## FFNN
        # self._fc1 = nn.Linear(768, 256)
        # self._dropout = nn.Dropout(p=0.1)
        #self._head = nn.Linear(256, num_labels)

        self._head = nn.Linear(768, num_labels)
        
    def forward(self, batch, mask):
        b = self._bert(batch)
        pooler = b.last_hidden_state[:, mask].diagonal().permute(2, 0, 1)

        ## FFNN
        # output = self._fc1(pooler)
        # f = self._dropout(output)
        # f = F.relu(f)
        #return self._head(f)
        return self._head(pooler)

def collate_fn(batch):
    longest_y = max([y.size(0) for X, y in batch])
    X = [X for X, y in batch]
    y = torch.stack([F.pad(y, (0, longest_y - y.size(0)), value=-1) for X, y in batch])
    return X, y

def build_mask(tokenizer, ids):
    tok_sents = [tokenizer.convert_ids_to_tokens(i) for i in ids]
    mask = []
    l = tokenizer.all_special_tokens
    l.remove('[UNK]')
    for sentence in tok_sents:
        current = []
        for n, token in enumerate(sentence):
            if token in l or token.startswith('##'):
                continue
            else:
                current.append(n)
        mask.append(current)

    mask = tokenizer.pad({'input_ids': mask}, return_tensors='pt')['input_ids']
    return mask

def create_class_weight(labels_dict,mu=0.15):
    total = np.sum(list(labels_dict.values()))
    keys = labels_dict.keys()
    class_weight = dict()
    
    for key in keys:
        score = math.log(mu*(total/float(labels_dict[key])))
        class_weight[key] = score if score > 1.0 else 1.0
    
    return class_weight

if __name__ == "__main__":
    data = parse(open("/cluster/projects/nn9851k/IN5550/norne-nb-in5550-train.conllu", "r").read())

    train_df, val_df = train_test_split(data, test_size = 0.1)
  
    train_dataset = ConnluDataset(train_df)    
    val_dataset = ConnluDataset(val_df, label_vocab=train_dataset.label_vocab)

    ##class weight calculations
    count_label = dict(Counter(train_dataset.flat))
    w = create_class_weight(count_label)
    w['O'] = 0.1
    c = train_dataset.label_indexer
    wl = [w[k] for k in c]
    weights = torch.FloatTensor(wl)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    norbert = "/cluster/shared/nlpl/data/vectors/latest/216/"
    tokenizer = transformers.BertTokenizer.from_pretrained(norbert)
    model = VecModel(norbert = norbert, num_labels = len(train_dataset.label_vocab))
    criterion = nn.CrossEntropyLoss(ignore_index=-1, weight=weights)

    ##selective regularization
    # no_decay = ['bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    # {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    # {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    # optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, eps=1e-8)

    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

    #selective fine-tuning
    #optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5, eps=1e-8)

    max_stagnation = 5
    best_val_loss, best_val_epoch = None, None
    early_stop = False

    train_loss_values, validation_loss_values = [], []
    for epoch in range(100):
        model.train()
        for X, y in tqdm.tqdm(train_loader):
            optimizer.zero_grad()
            X = tokenizer(X, is_split_into_words=True, return_tensors='pt', padding=True)['input_ids']
            batch_mask = build_mask(tokenizer, X)
            y_pred = model(X, batch_mask)
            y_pred = y_pred.permute(0, 2, 1)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

        model.eval()
        predictions , true_labels = [], []
        for X, y in tqdm.tqdm(val_loader):
            X = tokenizer(X, is_split_into_words=True, return_tensors='pt', padding=True)['input_ids']
            batch_mask = build_mask(tokenizer, X)
            y_pred = model(X, batch_mask)
            y_pred = y_pred.permute(0, 2, 1)
            val_loss = criterion(y_pred, y)
            
            x = y_pred.argmax(dim=1).detach().numpy()
            y = y.numpy()
            predictions.extend([list(p) for p in x])
            true_labels.extend(list(p) for p in y)

        if best_val_loss is None or best_val_loss < val_loss:
            best_val_loss, best_val_epoch = val_loss, epoch
        if best_val_epoch < epoch - max_stagnation:
            early_stop = True 
            print('Early stop.')
            break

        pred_tags = [train_dataset.inverse_indexer[p_i] for p, l in zip(predictions, true_labels)
                                         for p_i, l_i in zip(p, l) if l_i!= -1]
        valid_tags = [train_dataset.inverse_indexer[l_i] for l in true_labels
                                          for l_i in l if l_i!= -1]

        train_loss_values.append(loss.item())
        validation_loss_values.append(val_loss.item())
        print(f"epoch: {epoch}; train loss: {loss.item()}; val loss = {val_loss.item()}")
        print(f"f1 : {f1_score(pred_tags, valid_tags, average='weighted')}")

    print(classification_report(valid_tags, pred_tags))

    # #Increase the plot size and font size.
    # sns.set(font_scale=1.5)
    # plt.rcParams["figure.figsize"] = (12,6)

    # #Plot the learning curve.
    # plt.plot(train_loss_values, 'b-o', label="training loss")
    # plt.plot(validation_loss_values, 'r-o', label="validation loss")

    # #Label the plot.
    # plt.title("Learning curve")
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.legend()

    # plt.savefig("loss_plot.png", dpi = 300)

    #torch.save({'model': model.state_dict(),
    #            'label_vocab': train_dataset.label_vocab, 'label_indexer': train_dataset.label_indexer, 'inverse_indexer': train_dataset.inverse_indexer}, f"basic_model.pt")
