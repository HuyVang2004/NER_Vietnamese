import torch 
from torch.utils.data import Dataset
from typing import List, Dict

class NERDataset(Dataset):
    """
    data: list of {"words": [...], "tags": [...]}
    word2idx: Vocab.word2idx
    tag2idx: dict
    """
    def __init__(self, data: List[Dict], word2idx: function, tag2idx: dict, max_word_len = 20):
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
    def __init__(self, data: List[Dict], tokenizer, tag2idx: dict, max_seq_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.tag2idx = tag2idx
        self.max_seq_len = max_seq_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        words = sample['words']
        tags = sample['tags']

        encoding = self.tokenizer(words,
                                  is_split_into_words=True,
                                  return_offsets_mapping=True,
                                  padding='max_length',
                                  truncation=True,
                                  max_length=self.max_seq_len)

        word_ids = encoding.word_ids()
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        tidx = [self.tag2idx[t] for t in tags]

        out = {
            "widx": torch.tensor(input_ids, dtype=torch.long),
            "mask": torch.tensor(attention_mask, dtype=torch.long),
            "word_idx": word_ids,
            "tidx": tidx,
            "words": words
        }
        return out
    

