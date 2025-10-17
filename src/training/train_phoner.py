import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from seqeval.metrics import classification_report, f1_score
from src.utils.evaluate import evaluate
from transformers import AdamW, get_linear_schedule_with_warmup
from src.data.collate import collate_phobert


def evaluate_model(model, dataloader, device, id2label, use_crf=True):
    model.eval()
    all_preds = []
    all_golds = []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)  # (B, L) with -100 for ignored tokens

            # compute loss if model returns loss when labels provided
            if use_crf:
                loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_loss += loss.item()
                n_batches += 1
                # decode
                paths = model(input_ids=input_ids, attention_mask=attention_mask, labels=None)
                # convert preds & golds to BIO strings
                for i, path in enumerate(paths):
                    # length of valid tokens = number of tokens where labels != -100
                    valid_len = (labels[i] != -100).sum().item()
                    # path is a list of tag ids (for valid positions)
                    pred_labels = [id2label[t] for t in path]
                    gold_labels = []
                    # need to collect the gold labels for the same valid positions
                    lab_ids = labels[i][:]  # (L,)
                    gold_labels = [id2label[int(lab_ids[j])] for j in range(len(lab_ids)) if lab_ids[j] != -100]
                    all_preds.append(pred_labels)
                    all_golds.append(gold_labels)
            else:
                # non-CRF: forward returns loss if labels provided
                loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_loss += loss.item()
                n_batches += 1
                logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=None)  # (B, L, C)
                preds = torch.argmax(logits, dim=-1)  # (B, L)
                for i in range(preds.size(0)):
                    pred_i = preds[i]
                    lab_i = labels[i]
                    pred_labels = []
                    gold_labels = []
                    for j in range(pred_i.size(0)):
                        if lab_i[j].item() == -100:
                            continue
                        pred_labels.append(id2label[pred_i[j].item()])
                        gold_labels.append(id2label[lab_i[j].item()])
                    all_preds.append(pred_labels)
                    all_golds.append(gold_labels)

    mean_loss = total_loss / max(n_batches, 1)
    try:
        report = classification_report(all_golds, all_preds, digits=4)
        f1 = f1_score(all_golds, all_preds)
    except Exception:
        # fallback accuracy
        total = 0
        correct = 0
        for p, g in zip(all_preds, all_golds):
            for a, b in zip(p, g):
                total += 1
                if a == b:
                    correct += 1
        f1 = correct / total if total > 0 else 0.0
        report = "seqeval not available"
    return mean_loss, f1, report


def train_loop(model, train_dataset, dev_dataset, test_dataset,
               device, idx2tag, out_dir: str, num_epochs: int = 5, batch_size: int = 8,
               lr: float = 5e-5, weight_decay: float = 0.0, use_crf: bool = True):
    os.makedirs(out_dir, exist_ok=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=lambda b: collate_phobert(b))
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=lambda b: collate_phobert(b))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=lambda b: collate_phobert(b))

    # optimizer & scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps),
                                                num_training_steps=total_steps)

    best_dev_f1 = 0.0
    best_model_path = None

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

            if step % 50 == 0:
                avg = running_loss / step
                print(f"[Epoch {epoch}] Step {step}/{len(train_loader)} - avg loss: {avg:.4f}")

        avg_epoch_loss = running_loss / len(train_loader)
        dev_loss, dev_f1, dev_report = evaluate_model(model, dev_loader, device, idx2tag, use_crf=use_crf)
        print(f"Epoch {epoch} finished. Train loss: {avg_epoch_loss:.4f} | Dev loss: {dev_loss:.4f} | Dev F1: {dev_f1:.4f}")
        print(dev_report)

        # save best
        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_model_path = os.path.join(out_dir, f"best_model_epoch{epoch}.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f">>> Best model saved to {best_model_path}")

    # final test with best modelid2tag = {i:t for t,i in tag2idx.items()}
    if best_model_path:
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_loss, test_f1, test_report = evaluate_model(model, test_loader, device, idx2tag, use_crf=use_crf)
    print(f"\nFinal Test - loss: {test_loss:.4f} | F1: {test_f1:.4f}")
    print(test_report)
    return model
