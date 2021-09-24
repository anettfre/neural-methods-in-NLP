import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from collections import Counter
import time
import pickle

### Seed for reproducibility
torch.manual_seed(1337)

### Our model with the best parameters, implemented as a class
### Using dropout to reduce overfitting
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

### Fuction to load the file, create BoW vector with a max vocabulary and divide in train and test/dev set
def datasetloader(filename, vocab_size=2000):

    data_start= time.time()
    df = pd.read_csv(filename, sep = '\t', header = 0, compression = 'gzip')

    words = df['text']
    classes = df['source']

    ### To get BoW vector with at most vocab_size words
    vectorizer = CountVectorizer(max_features = vocab_size, ngram_range = (1,2))
    ### To get classes in 0-19 format
    encoder = LabelEncoder()
    
    in_features = vectorizer.fit_transform(words.values).toarray().astype(np.float32)
    #with open('../../vectorizer.pickle', 'wb') as f1:
    #    pickle.dump(vectorizer, f1)

    labels = encoder.fit_transform(classes.values)
    #with open('encoder.pickle', 'wb') as f2:
    #    pickle.dump(encoder, f2)


    ### Dividing in train and test/dev set. With 80% of data in train.
    ### using stratify to divide equaly amount of classes in each set.  
    X_train, X_test, y_train, y_test = train_test_split(in_features, labels, test_size = 0.2, stratify = labels)

    ### Making the sets into PyTorch objects 
    X_train, X_test = torch.from_numpy(X_train).type(torch.float), torch.from_numpy(X_test).type(torch.float)
    y_train, y_test = torch.from_numpy(y_train).type(torch.long), torch.from_numpy(y_test).type(torch.long)

    ### Creating PyTorch dataset objects for train and test
    train = TensorDataset(X_train, y_train)
    test = TensorDataset(X_test, y_test)
    
    ### Creating batches, using 8 samples in train and the whole set for test
    train_batch = DataLoader(train, batch_size=8, shuffle=True)
    test_batch = DataLoader(test, batch_size=len(test), shuffle=False)

    data_end = time.time()
    print("Data loaded and transformed in %s minutes" % ((data_end - data_start) // 60))

    return train_batch, test_batch

### Setting size of vocabulary, need it when creating model and BoW vector 
vocab_size = 2000

train, test = datasetloader('../../signal_20_obligatory1_train.tsv.gz', vocab_size)

model = Model(in_features=vocab_size)

### Loss function
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)

epochs = 50
losses = []

### Starting to train
start= time.time()
for i in range(epochs):
    loss = 0

    ### Looping over the batches in train and fitting the model for each
    for x, y in train:
        y_pred = model.forward(x)
      
        loss = criterion(y_pred,y)
        losses.append(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    ### Print out every 10 epoch on training loss
    if i%10 == 0:
        print(f'Epoch {i} out of {epochs} --> Loss: {loss}')
end = time.time()
print("Model finished training in %s minutes" % ((end - start) // 60))

### Lists for storing prediction and true values 
y_pred = []
y_true = []

### Evaluation on the test set for getting a realistic F1-score++
### Using no_grad for faster predictions and reduce memory usage
with torch.no_grad():
    for x, y in test:
        y_true.append(y)
        y_eval = model.forward(x)
        predictions = y_eval.max(dim=1)[1]
        y_pred.append(predictions)

    ### Print for evalutaion, y_pred and y_true is a nested list (therefore "[0]")
    print(accuracy_score(y_true[0], y_pred[0]))
    print(classification_report(y_true[0], y_pred[0], digits=4))

### Last line is saving the model
#torch.save(model.state_dict(), 'model_state.pth')
