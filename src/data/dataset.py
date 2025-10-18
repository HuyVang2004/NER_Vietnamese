import torch 
from torch.utils.data import Dataset
from typing import List, Dict

class NERDataset(Dataset):
    """
    data: list of {"words": [...], "tags": [...]}
    word2idx: Vocab.word2idx
    tag2idx: dict
    """
    def __init__(self, data: List[Dict], word2idx, tag2idx: dict, max_word_len = 20):
        self.data = data
        self.word2idx = word2idx
        self.tag2idx = tag2idx
        self.max_word_len = max_word_len

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        words = sample['words']
        tags = sample['tags']

        widx = [self.word2idx(w) for w in words]
        tidx = [self.tag2idx[t] for t in tags]

        out = {"widx": widx, "tidx": tidx, "words": words}
        return out
    

class NERDatasetPhoBERT(Dataset):
    def __init__(self, data: List[Dict], tokenizer, tag2idx: dict):
        self.data = data
        self.tokenizer = tokenizer
        self.tag2idx = tag2idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        words = sample["words"]
        tags = sample["tags"]

        # Tokenize từng word riêng lẻ để track alignment
        input_ids = []
        labels = []
        
        # Thêm token đặc biệt đầu (bos_token hoặc cls_token)
        input_ids.append(self.tokenizer.bos_token_id or self.tokenizer.cls_token_id)
        labels.append(-100)
        
        for word, tag in zip(words, tags):
            # Tokenize từng word
            word_tokens = self.tokenizer.encode(word, add_special_tokens=False)
            
            # Token đầu tiên của word nhận label
            if len(word_tokens) > 0:
                input_ids.extend(word_tokens)
                labels.append(self.tag2idx[tag])
                # Các subword token còn lại nhận -100
                labels.extend([-100] * (len(word_tokens) - 1))
        
        # Thêm token đặc biệt cuối (eos_token hoặc sep_token)
        input_ids.append(self.tokenizer.eos_token_id or self.tokenizer.sep_token_id)
        labels.append(-100)
        
        # Tạo attention mask
        attention_mask = [1] * len(input_ids)
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }