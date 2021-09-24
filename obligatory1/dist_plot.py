from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('signal_20_obligatory1_train.tsv.gz', sep='\t', header=0, compression='gzip')

train, test = train_test_split(df, test_size = 0.2, stratify = df['source'], shuffle = True)

uSTrain = train.source.unique()

train_s = []
train_l = []
test_s = []
test_l = []

for i in uSTrain:
	train2 = train.loc[train['source']==i]
	train_l.append(train2.shape[0])

for j in uSTrain:
	test2 = test.loc[test['source']==j]
	test_l.append(test2.shape[0])


p1 = plt.bar(uSTrain, train_l, color = 'blue')
p2 = plt.bar(uSTrain, test_l, bottom = train_l, color = 'red')
plt.xticks(rotation=45, horizontalalignment="right")
plt.tight_layout()
plt.show()

