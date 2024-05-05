import torch
from torch import nn 

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_dim, num_layers, dropout):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # input_size = len vocab of source (in this example is en)
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, dropout= dropout, batch_first= True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch_size, max_length] max_length is sentence length after padding
        embedding = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.rnn(embedding)
        return hidden, cell
