import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

from src.models.crf import CRF


class TransformerForTokenClassification(nn.Module):
    
    def __init__(self, pretrained_model_name: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.num_labels = num_labels

        cfg = AutoConfig.from_pretrained(pretrained_model_name, output_hidden_states=False)
        self.encoder = AutoModel.from_pretrained(pretrained_model_name, config=cfg)
        hidden_size = cfg.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

        # ignore_index=-100 để bỏ qua padding/subword
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, attention_mask, labels=None, return_logits=False):
        vocab_size = self.encoder.config.vocab_size
        num_labels = self.num_labels

        
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(hidden)  # [B, L, num_labels]

        loss = None
        if labels is not None:
            # labels shape [B, L]
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        preds = torch.argmax(logits, dim=-1)

        if return_logits:
            return preds, logits, loss
        return preds if loss is None else (preds, loss)