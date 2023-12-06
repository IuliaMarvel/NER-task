import pytorch_lightning as pl
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

class plNERModel(pl.LightningModule):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_size, lr):
        super(plNERModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.output_size = output_size
        self.fc = nn.Linear(hidden_dim * 2, output_size)
        self.criterion = nn.CrossEntropyLoss()
        self.training_history = []
        self.validation_history = []
        self.lr = lr

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x) # lstm_out_shape: Batchsize x SeLength x (D * Hidden)
        output = self.fc(lstm_out)
        return output

    def training_step(self, batch, idx):
        # training_step defines the train loop.
        x, labels = batch
        outputs = self.forward(x)
        outputs = outputs.view(-1, self.output_size)
        labels = labels.view(-1)
        loss = self.criterion(outputs, labels)
        self.training_history.append(loss.item())
        return loss

    def validation_step(self, batch, idx):
        x, labels = batch
        outputs = self.forward(x)
        outputs = outputs.view(-1, self.output_size)
        labels = labels.view(-1)
        loss = self.criterion(outputs, labels)
        self.validation_history.append(loss.item())
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


    def plot_training(self, epochs):
        for phase in ['train', 'test']:
            if phase == 'train':
                history = self.training_history
            else:
                history = self.validation_history
            batch_size = len(history) // epochs
            loss_history = [np.mean(history[i : i + batch_size]) for i in range(0, epochs * batch_size, batch_size)]
            plt.plot(range(1, len(loss_history) + 1), loss_history, label=phase)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title(f'Losses')
        plt.savefig(f'Loss_history.png')


def get_model(vocab_size, embedding_dim, hidden_dim, output_size, lr):
    model = plNERModel(vocab_size, embedding_dim, hidden_dim, output_size, lr)
    return model