import torch
from torch import nn
import random

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert (
            encoder.hidden_dim == decoder.hidden_dim
        ), "Hidden dimensions of encoder and decoder must be equal!"
        assert (
            encoder.num_layers == decoder.num_layers
        ), "Encoder and decoder must have equal number of layers!"

    def forward(self, source, target, teacher_force_ratio= 0.5) -> torch.Tensor:
        # source: [batch_size, src_length]
        # target: [batch_size, tar_length]
        # Note: src_length = tar_length = max_length (padding length)

        batch_size = source.shape[0]
        target_len = target.shape[1]
        target_vocab_size = self.decoder.output_size

        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(self.device)
        # outputs: [batch_size, target_len, target_vocab_size]

        hidden, cell = self.encoder(source)

        # start token (usually <sos>)
        x = target[:, 0]

        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell)

            outputs[:,t,:] = output

            best_guess = output.argmax(1)

            x = target[:, t] if random.random() < teacher_force_ratio else best_guess
        
        return outputs