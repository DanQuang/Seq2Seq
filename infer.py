import torch
from torch import nn, optim
import os
from tqdm.auto import tqdm
from model import Seq2Seq
from encoder import Encoder
from decoder import Decoder
from load_data import Load_Data

class Train_Task:
    def __init__(self, config):
        self.num_epochs = config["num_epochs"]
        self.learning_rate = config["learning_rate"]
        self.save_path = config["save_path"]
        self.patience = config["patience"]
        self.dropout = config["dropout"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load text embedding config
        self.embedding_dim = config["text_embedding"]["embedding_dim"]
        self.max_length = config["text_embedding"]["max_length"]

        # Load model config
        self.hidden_dim = config["model"]["hidden_units"]
        self.num_layers = config["model"]["num_layers"]

        # Load data
        self.data_path = config["data_path"]
        self.dataloader = Load_Data(self.data_path, self.max_length)

        # Load vocab
        self.vocab_en = self.dataloader.vocab_en
        self.vocab_de = self.dataloader.vocab_de

        # Get vocab size of 2 vocab
        en_vocab_size = self.vocab_en.vocab_size()
        de_vocab_size = self.vocab_de.vocab_size()

        # Load Encoder, Decoder
        self.encoder = Encoder(
            input_size= en_vocab_size,
            embedding_dim= self.embedding_dim,
            hidden_dim= self.hidden_dim,
            num_layers= self.num_layers,
            dropout= self.dropout
        )
        self.decoder = Decoder(
            output_size= de_vocab_size,
            embedding_dim= self.embedding_dim,
            hidden_dim= self.hidden_dim,
            num_layers= self.num_layers,
            dropout= self.dropout
        )

        # Load model
        self.model = Seq2Seq(self.encoder, self.decoder, self.device).to(self.device)

    def predict(self):
        test = self.dataloader.load_test()

        best_model = "Seq2Seq_best_model.pth"

        if os.path.join(self.save_path, best_model):
            checkpoint = torch.load(os.path.join(self.save_path, best_model))
            self.model.load_state_dict(checkpoint["model_state_dict"])

            
            self.model.eval()
            with torch.inference_mode():
                epoch_loss = 0
                for _, item in enumerate(tqdm(test)):
                    source, target = item["en_ids"].to(self.device), item["de_ids"].to(self.device)
                    output = self.model(source, target, 0) # turn off teacher fourcing
                    # outputs: [batch_size, target_len, target_vocab_size]
                    output_dim = output.shape[-1] # target_vocab_size

                    output = output[:, 1:, :].view(-1, output_dim)
                    # output: [batch_size*(target_len - 1), target_vocab_size]

                    target = target[:, 1:].view(-1)
                    # target: [batch_size*(target_len - 1)]

                    loss = self.criterion(output, target)

                    epoch_loss += loss.item()

                test_loss = epoch_loss / len(test)

                print(f"Test loss: {test_loss:.5f}")
