import torch
from torch import nn 

class Decoder(nn.Module):
    def __init__(self, output_size, embedding_dim, hidden_dim, num_layers, dropout):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # output_size = len vocab of target
        self.output_size = output_size
        self.embedding = nn.Embedding(output_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, dropout= dropout, batch_first= True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x, hidden, cell):
        # x: [batch_size]
        x = x.unsqueeze(1)
        # x: [batch_size, 1]

        embedding = self.dropout(self.embedding(x))
        # embedding: [batch_size, 1, embedding_dim]

        outputs, (new_hidden, new_cell) = self.rnn(embedding, (hidden, cell))
        # output: [batch_size, 1, hidden_dim]

        outputs = outputs.squeeze(1)
        pred = self.fc(outputs)
        # pred: [batch_size, output_size]
        return pred, new_hidden, new_cell