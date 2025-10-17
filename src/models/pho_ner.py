import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

from src.models.crf import CRF


class TransformerForTokenClassification(nn.Module):
    """
    Transformer for token classification (NER).
    Supports optional CRF on top.

    Inputs expected (during training/inference):
        - input_ids: [B, L]
        - attention_mask: [B, L]
        - token_mask: [B, L]  -> boolean: True at positions corresponding to first subword of words
        - labels: [B, L] -> for CRF: tag ids (any value where token_mask==False), for CE: use -100 to ignore
    """
    def __init__(self, pretrained_model_name: str, num_labels: int, use_crf: bool = False, dropout: float = 0.1, pad_tag_id: int = 0):
        super().__init__()
        self.num_labels = num_labels
        self.use_crf = use_crf
        self.pad_tag_id = pad_tag_id

        cfg = AutoConfig.from_pretrained(pretrained_model_name, output_hidden_states=False)
        self.encoder = AutoModel.from_pretrained(pretrained_model_name, config=cfg)
        hidden_size = cfg.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

        if use_crf:
            self.crf = CRF(num_tags=num_labels, pad_idx=pad_tag_id)
        else:
            self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, attention_mask, token_mask=None, labels=None):
        """
        If labels provided:
            - if use_crf: returns loss (scalar tensor)
            - else: returns loss (scalar tensor)
        If labels not provided:
            - if use_crf: returns list of best tag sequences (list of lists)
            - else: returns logits (B,L,T) or argmax predictions
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [B, L, H]
        sequence_output = self.dropout(sequence_output)
        emissions = self.classifier(sequence_output)  # [B, L, num_labels]

        if labels is not None:
            if self.use_crf:
                # For CRF, pass tags and mask where token_mask==1
                if token_mask is None:
                    # default: use attention_mask
                    mask = attention_mask.bool()
                else:
                    mask = token_mask.bool()
                loss = self.crf(emissions, labels, mask)
                return loss
            else:
                # For CE: set labels to -100 where token_mask==0
                if token_mask is not None:
                    labels_ce = labels.clone()
                    labels_ce[~token_mask.bool()] = -100
                else:
                    labels_ce = labels
                B, L, C = emissions.size()
                logits = emissions.view(-1, C)
                loss = self.loss_fct(logits, labels_ce.view(-1))
                return loss
        else:
            # inference
            if self.use_crf:
                if token_mask is None:
                    mask = attention_mask.bool()
                else:
                    mask = token_mask.bool()
                paths = self.crf.viterbi_decode(emissions, mask)
                return paths
            else:
                # return logits (caller can argmax)
                return emissions