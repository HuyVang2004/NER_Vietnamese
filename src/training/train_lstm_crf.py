import torch
import torch.optim as optim
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from torch.utils.data import DataLoader
from tqdm import tqdm
from src.utils.evaluate import evaluate
from src.data.collate import collate_dl
import os

def train_epoch(model, dataloader, optimizer, device, pad_tag_id=0):
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader):
        wids = batch["widx"].to(device)
        tids = batch["tidx"].to(device)
        mask = batch["mask"].to(device)
       
        optimizer.zero_grad()
        loss = model(widx=wids, mask=mask, tags=tids)  
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def train(model, train_loader, dev_loader, optimizer, 
           id2tag, device, num_epochs=10, use_crf=True):

    for epoch in range(1, num_epochs + 1):
        model.train()
        mean_loss = train_epoch(model, train_loader, optimizer, device, pad_tag_id=0)

        dev_loss, dev_f1, dev_report = evaluate(model, dev_loader, device, id2tag, use_crf)
        print(f"Epoch {epoch}: Train Loss={mean_loss:.4f}, Dev Loss={dev_loss:.4f}, Dev F1={dev_f1:.4f}")
        print("Dev Classification Report:\n", dev_report)


def train_loop(model, train_dataset, dev_dataset, test_dataset, 
               device, id2tag, out_dir: str, num_epochs: int = 10, 
               batch_size: int = 32, lr: float = 1e-3, 
               weight_decay: float = 0.0, use_crf: bool = True, collate_fn=None):
  
    os.makedirs(out_dir, exist_ok=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn)

    # Optimizer
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    #
    best_f1 = train(model, train_loader, dev_loader, optimizer, device, id2tag, num_epochs, use_crf)

    # Evaluate best model
    print("Evaluating on test set...")
    model.load_state_dict(torch.load("best_model.pt"))
    test_loss, test_f1, test_report = evaluate(model, test_loader, device, id2tag, use_crf)
    print(f"Test F1 = {test_f1:.4f}")
    print(test_report)
