import torch
from torch import nn
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self, args, num_labels, embeddings_model):
        super().__init__()
        self.hidden_dim = args.hidden_dim
        self.embeddingbag = nn.EmbeddingBag.from_pretrained(torch.FloatTensor(embeddings_model.wv.vectors), mode="sum")
        self._hidden = nn.Linear(embeddings_model.vector_size, args.hidden_dim)
        self._output = nn.Linear(args.hidden_dim, num_labels)
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, X):
        embedding = self.embeddingbag(X, torch.LongTensor([0]))
        hidden = self._hidden(embedding)
        hidden = self.dropout(hidden)
        hidden = F.relu(hidden)
        output = self._output(hidden)
        
        return output
