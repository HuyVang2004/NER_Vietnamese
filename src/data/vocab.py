from collections import Counter, defaultdict
import json

class Vocab:
    def __init__(self, min_freq=1, unk_token="<UNK>", pad_token="<PAD>"):
        self.min_freq = min_freq
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.freqs = Counter()
        self.sen2idx = {}
        self.idx2sen = []

    def build_from_sentences(self, sentences):
        for sent in sentences:
            for w in sent:
                self.freqs[w] += 1

        self.idx2sen = [self.pad_token, self.unk_token]
        for w, f in self.freqs.items():
            if f >= self.min_freq:
                self.idx2sen.append(w)
        
        self.sen2idx = {w:i for i, w in enumerate(self.idx2sen)}

    def __len__(self):
        return len(self.idx2sen)
    
    def word2idx(self, w):
        return self.sen2idx.get(w, self.sen2idx[self.unk_token])
    
    def idx2word(self, idx):
        return self.idx2sen[idx]
    
    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.idx2sen, f, ensure_ascii=False)

    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            self.idx2sen = json.load(f)

        self.sen2idx = {w:i for i, w in enumerate(self.idx2sen)}