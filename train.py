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

        self.optim = optim.Adam(self.model.parameters(), lr= self.learning_rate)
        self.criterion = nn.CrossEntropyLoss(ignore_index= self.vocab_de.pad_idx())

    def train(self):
        train, dev = self.dataloader.load_train_dev()

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        last_model = "Seq2Seq_last_model.pth"
        best_model = "Seq2Seq_best_model.pth"

        if os.path.exists(os.path.join(self.save_path, last_model)):
                checkpoint = torch.load(os.path.join(self.save_path, last_model))
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.optim.load_state_dict(checkpoint["optim_state_dict"])
                print("Load the last model")
                initial_epoch = checkpoint["epoch"] + 1
                print(f"Continue training from epoch {initial_epoch}")
        else:
            initial_epoch = 0
            print("First time training!!!")

        if os.path.exists(os.path.join(self.save_path, best_model)):
            checkpoint = torch.load(os.path.join(self.save_path, best_model))
            best_score = checkpoint['score']
        else:
            best_score = 0.

        threshold = 0

        self.model.train()
        for epoch in range(initial_epoch, initial_epoch + self.num_epochs):
            epoch_loss = 0
            for _, item in enumerate(tqdm(train)):
                source, target = item["en_ids"].to(self.device), item["de_ids"].to(self.device)

                self.optim.zero_grad()

                output = self.model(source, target)
                # outputs: [batch_size, target_len, target_vocab_size]
                output_dim = output.shape[-1] # target_vocab_size

                output = output[:, 1:, :].view(-1, output_dim)
                # output: [batch_size*(target_len - 1), target_vocab_size]

                target = target[:, 1:].view(-1)
                # target: [batch_size*(target_len - 1)]

                loss = self.criterion(output, target)

                loss.backward()

                self.optim.step()

                epoch_loss += loss.item()
            
            train_loss = epoch_loss / len(train)
            print(f"Epoch {epoch}:")
            print(f"Train loss: {train_loss:.5f}")

            epoch_loss = 0
            with torch.inference_mode():
                for _, item in enumerate(tqdm(dev)):
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

                valid_loss = epoch_loss / len(dev)

                print(f"Valid loss: {valid_loss:.5f}")

                score = valid_loss

                # save last model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optim_state_dict': self.optim.state_dict(),
                    'score': score
                }, os.path.join(self.save_path, last_model))

                # save the best model
                if epoch > 0 and score < best_score:
                    threshold += 1
                else:
                    threshold = 0

                if score > best_score:
                    best_score = score
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optim_state_dict': self.optim.state_dict(),
                        'score':score
                    }, os.path.join(self.save_path, best_model))
                    print(f"Saved the best model with valid loss of {score:.5f}")
