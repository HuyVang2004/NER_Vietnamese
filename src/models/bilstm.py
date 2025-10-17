import torch 
import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=200,
                 hidden_dim=256, pretrained_emb=None, dropout=0.3, pad_idx=0):
        super.__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        if pretrained_emb is not None:
            self.embedding.weight.data.copy_(pretrained_emb)
        total_emb_dim = embedding_dim

        self.bilstm = nn.LSTM(total_emb_dim, hidden_dim//2, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, widx, mask):
        emb = self.embedding(widx)
        lengths = mask.sum(dim=1).cpu()

        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.bilstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)  # [B, L, hidden_dim]
        out = self.dropout(out)
        emissions = self.hidden2tag(out)  # [B, L, tagset]
        
        return emissions