import torch
from tqdm import tqdm
from seqeval.metrics import classification_report, f1_score

def evaluate(model, dataloader, device, id2tag, use_crf=True):
    model.eval()
    preds = []
    golds = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            wids = batch["widx"].to(device)
            tids = batch["tidx"].to(device)
            mask = batch["mask"].to(device)
            
            if use_crf:
                loss = model(wids, mask, tags=tids)
                batch_loss = loss.mean().item() if loss.dim() > 0 else loss.item()
                total_loss += batch_loss
                total_batches += 1

                batch_paths = model(wids, mask, tags=None)  # list of paths
                
                # map to tags (strings)
                for i, path in enumerate(batch_paths):
                    pred_tags = [id2tag[t] for t in path]
                    gold_len = mask[i].sum().item()
                    gold_tags = [id2tag[t.item()] for t in tids[i, :gold_len]]
                    preds.append(pred_tags)
                    golds.append(gold_tags)
            else:
                emissions = model.bilstm(wids, mask)  # [B,L,T]
                loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

                # reshape for CE: (B*L, T)
                active_tokens = mask.view(-1) == 1
                logits = emissions.view(-1, emissions.shape[-1])
                labels = tids.view(-1)

                if active_tokens.sum() > 0:
                    active_logits = logits[active_tokens]
                    active_labels = labels[active_tokens]
                    batch_loss = loss_fn(active_logits, active_labels).item()
                    total_loss += batch_loss
                    total_batches += 1

                _, max_tags = emissions.max(dim=-1)  # [B,L]
                for i in range(wids.size(0)):
                    gold_len = mask[i].sum().item()
                    pred_tags = [id2tag[t.item()] for t in max_tags[i, :gold_len]]
                    gold_tags = [id2tag[t.item()] for t in tids[i, :gold_len]]
                    preds.append(pred_tags)
                    golds.append(gold_tags)
 
    mean_loss = total_loss / max(total_batches, 1)
    try:
        report = classification_report(golds, preds, digits=4)
        f1 = f1_score(golds, preds)
    except Exception:
        total, correct = 0, 0
        for p, g in zip(preds, golds):
            for a, b in zip(p, g):
                total += 1
                if a == b:
                    correct += 1
        f1 = correct / total if total > 0 else 0.0
        report = "seqeval not installed, fallback accuracy used"

    return mean_loss, f1, report