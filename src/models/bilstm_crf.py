import torch
import torch.nn as nn
from src.models.bilstm import BiLSTM
from src.models.crf import CRF

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=200, hidden_dim=256,
                 pretrained_embeddings=None, pad_idx=0):
        super().__init__()
        self.bilstm = BiLSTM(vocab_size, tagset_size, embedding_dim, hidden_dim,
                                   pretrained_emb=pretrained_embeddings,
                                   pad_idx=pad_idx)
        self.crf = CRF(num_tags=tagset_size, pad_idx=pad_idx)

    def forward(self, wids, mask, tags=None):
        """
        If tags provided -> compute loss (negative log-likelihood)
        If tags not provided -> decode best path
        """
        emissions = self.bilstm(wids, mask)  # [B, L, tagset]
        if tags is not None:
            loss = self.crf(emissions, tags, mask)
            return loss
        else:
            return self.crf.viterbi_decode(emissions, mask)
