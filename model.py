import json
from turtle import forward
import torch
from torch import nn
from poem_dataset import PoemDataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from torch.optim.adam import Adam

class Encoder(nn.Module):
    def __init__(self, inp_size, hid_size):
        super(Encoder, self).__init__()
        self.BiLSTM = nn.GRU(inp_size, hid_size, num_layers=3)
        self.Linear = nn.Linear(hid_size, hid_size)

    def forward(self, inp):
        out, hid = self.BiLSTM(inp)
        out = self.Linear(out)
        out = torch.sigmoid(out)
        return out, hid


class Attention(nn.Module):
    def __init__(self, inp_size, hid_size) -> None:
        super(Attention, self).__init__()
        self.BiLSTM = nn.LSTM(inp_size, hid_size, num_layers=3, bidirection=True)
    
    def forward(self, inp):
        out, (hid, cel) = self.BiLSTM(inp)

class Decoder(nn.Module):
    def __init__(self, inp_size, hid_size, window_size, word_size):
        super(Decoder, self).__init__()
        self.inp_size = inp_size
        self.GRU = nn.GRU(inp_size, hid_size, num_layers=1, batch_first=True)
        self.Linear = nn.Linear(hid_size, word_size)

    def forward(self, inp):
        inp = inp.reshape(-1, self.inp_size)
        _out, hid = self.GRU(inp)
        _out = self.Linear(_out)
        out = torch.sigmoid(_out)
        return out

class PoemGeneratorModel(nn.Module):
    def __init__(self, word_size, embed_size, hid_size, window_size) -> None:
        super().__init__()
        self.Encoder = Encoder(embed_size, hid_size)
        self.sentiment = nn.Linear(embed_size, hid_size)
        self.scale = nn.Linear(hid_size * 2, hid_size)
        self.Decoder = Decoder(embed_size, hid_size, window_size, word_size)
    
    def forward(self, input, sen):
        out = self.Decoder(input)
        return out

class PoemGeneratorLightning(pl.LightningModule):
    def __init__(self, word_size, embed_size, hid_size, window_size, seq_len, batch_size=32, lr=0.0001, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = PoemGeneratorModel(word_size, embed_size, hid_size, window_size)
        self.word_size = word_size
        self.seq_len = seq_len
        self.embed_size = embed_size
        self.hid_size = hid_size
        self.batch_size = batch_size
        self.lr = lr
    
    def forward(self, inp, att):
        return self.model(inp, att)

    def setup(self, stage: str):
        inp_ = "/data/vectorized/inp.pkl"
        ctr_ = "/data/vectorized/inp.pkl"
        out_ = "/data/vectorized/out.pkl"
        dataset = PoemDataset(inp_, ctr_, out_)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        self.train_set, self.val_set = random_split(dataset, [train_size, val_size])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)
    
    def training_step(self, batch, batch_idx):
        inp, att, out = batch
        predict = self(inp, att)
        loss = F.cross_entropy(predict, out)
        with open('./tmp/inp.json', 'w') as f:
            json.dump({
                "inp": inp.size(),
                "pre": predict.size(),
                "out": out.size(),
                "loss": loss.tolist()
            }, f)
        return loss

    def validation_step(self, batch, batch_idx):
        inp, att, out = batch
        predict = self(inp, att)
        val_loss = F.cross_entropy(predict, out)
        self.log('val_loss', val_loss)