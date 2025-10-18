# bilstm_crf.py
import torch
import torch.nn as nn
from src.models.bilstm import BiLSTM
from src.models.crf import CRF


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=200,
                 hidden_dim=256, pretrained_emb=None, dropout=0.3, pad_idx=0):
        super().__init__()
        self.bilstm = BiLSTM(vocab_size, tagset_size, embedding_dim, hidden_dim,
                             pretrained_emb, dropout, pad_idx)
        self.crf = CRF(tagset_size, pad_idx)

    def forward(self, widx, tags, mask):
        emissions = self.bilstm(widx, mask)

        if tags is not None:
            loss = self.crf(emissions, tags, mask)
            return loss
        else:
            paths = self.crf.decode(emissions, mask)
            return paths

    def predict(self, widx, mask):
        emissions = self.bilstm(widx, mask)
        paths = self.crf.decode(emissions, mask)
        return paths
