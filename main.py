import torch
from torch.utils.data import DataLoader

from src.data.collate import collate_dl
from src.data.dataset import NERDataset
from data.vocab import Vocab
from src.utils.io_data import load_json_data
from src.models.bilstm_crf import BiLSTM_CRF
from src.training.train_lstm_crf import train, evaluate


train_path = "data/raw/PhoNER_COVID19/word/train_word.json"
test_path = "data/raw/PhoNER_COVID19/word/test_word.json"
dev_path = "data/raw/PhoNER_COVID19/word/dev_word.json"

train_data = load_json_data(train_path)
dev_data = load_json_data(dev_path)
test_data = load_json_data(test_path)

vocab = Vocab(min_freq=1)
vocab.build_from_sentences([sample['words'] for sample in train_data])

tagset = sorted({t for s in train_data for t in s["tags"]})
tag2idx = {t:i for i,t in enumerate(tagset)}
idx2tag = {i:t for t,i in tag2idx.items()}

train_dataset = NERDataset(train_data, vocab.word2idx, tag2idx)
dev_dataset = NERDataset(dev_data, vocab.word2idx, tag2idx)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=lambda b: collate_dl(b, pad_token_id=vocab.sen2idx[vocab.pad_token], pad_tag_id=tag2idx["O"]))
dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False, collate_fn=lambda b: collate_dl(b, pad_token_id=vocab.sen2idx[vocab.pad_token], pad_tag_id=tag2idx["O"]))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BiLSTM_CRF(vocab_size=len(vocab), tagset_size=len(tagset), embedding_dim=200, hidden_dim=256)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)

train(model, train_loader, dev_loader, optimizer, tag2idx, num_epochs=10, device=device)
