import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, RobertaTokenizerFast
from src.data.collate import collate_dl
from src.data.dataset import NERDataset, NERDatasetPhoBERT
from src.data.vocab import Vocab
from src.utils.io_data import load_json_data
from src.models.bilstm_crf import BiLSTM_CRF
from src.training.train_lstm_crf import train, evaluate
from src.models.pho_ner import TransformerForTokenClassification
from src.training.train_phoner import train_loop

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

# train_dataset = NERDataset(train_data, vocab.word2idx, tag2idx)
# dev_dataset = NERDataset(dev_data, vocab.word2idx, tag2idx)
# test_dataset = NERDataset(test_data, vocab.word2idx, tag2idx)
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=lambda b: collate_dl(b, pad_token_id=vocab.sen2idx[vocab.pad_token], pad_tag_id=tag2idx["O"]))
# dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False, collate_fn=lambda b: collate_dl(b, pad_token_id=vocab.sen2idx[vocab.pad_token], pad_tag_id=tag2idx["O"]))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = BiLSTM_CRF(vocab_size=len(vocab), tagset_size=len(tag2idx), embedding_dim=500, hidden_dim=256)
# model.to(device)
# optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)

# train(model, train_loader, dev_loader, optimizer, idx2tag, num_epochs=100, device=device, use_crf=True)
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base",
                                        use_fast=True,
                                        add_prefix_space=True)

# tokenizer = RobertaTokenizerFast.from_pretrained("vinai/phobert-base", add_prefix_space=True)

train_dataset = NERDatasetPhoBERT(train_data, tokenizer, tag2idx)
test_dataset = NERDatasetPhoBERT(test_data, tokenizer, tag2idx)
dev_dataset = NERDatasetPhoBERT(dev_data, tokenizer, tag2idx)

model = TransformerForTokenClassification(pretrained_model_name="vinai/phobert-base",
                    num_labels=len(tag2idx))
model.to(device)     
train_loop(model, tokenizer, train_dataset, dev_dataset, test_dataset,
               device=device, idx2tag=idx2tag, out_dir="./models", num_epochs = 10, batch_size = 16,
               lr = 5e-5)