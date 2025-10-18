# evaluate.py (replace existing evaluate with this)
from tqdm import tqdm
import torch
from seqeval.metrics import classification_report, f1_score

def evaluate(model, dataloader, device, id2tag, use_crf=True):
    model.eval()
    preds, golds = [], []
    total_loss, total_batches = 0.0, 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            wids = batch["widx"].to(device)
            tids = batch["tidx"].to(device)
            mask = batch["mask"].to(device)

            if use_crf:
                # model should return scalar NLL (positive)
                loss = model(widx=wids, mask=mask, tags=tids)
                # expect positive loss (NLL). If it's negative, there is bug in CRF.forward.
                batch_loss = loss.item() if isinstance(loss, torch.Tensor) else float(loss)
                total_loss += batch_loss
                total_batches += 1

                batch_paths = model(widx=wids, mask=mask, tags=None)  # list of lists

                for i, path in enumerate(batch_paths):
                    gold_len = int(mask[i].sum().item())
                    
                    pred_tags = [id2tag.get(int(t), "O") for t in path[:gold_len]]
                    gold_tags = [id2tag.get(int(t), "O") for t in tids[i, :gold_len]]
                    preds.append(pred_tags)
                    golds.append(gold_tags)
                    # print(id2tag)
                    # print(preds)
                    # print(golds)

            else:
                emissions = model.bilstm(wids, mask)  # [B,L,T]
                loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

                active_tokens = (mask.view(-1) == 1)
                logits = emissions.view(-1, emissions.shape[-1])
                labels = tids.view(-1)

                if active_tokens.sum() > 0:
                    active_logits = logits[active_tokens]
                    active_labels = labels[active_tokens]
                    batch_loss = loss_fn(active_logits, active_labels).item()
                    total_loss += batch_loss
                    total_batches += 1

                _, max_tags = emissions.max(dim=-1)
                for i in range(wids.size(0)):
                    gold_len = int(mask[i].sum().item())
                    pred_tags = [id2tag.get(int(t), "O") for t in max_tags[i, :gold_len]]
                    gold_tags = [id2tag.get(int(t), "O") for t in tids[i, :gold_len]]
                    preds.append(pred_tags)
                    golds.append(gold_tags)

    mean_loss = total_loss / max(total_batches, 1)

    try:
        report = classification_report(golds, preds, digits=4)
        f1 = f1_score(golds, preds)
    except Exception as e:
        print(f"[Warning] seqeval not available or error ({e}). Using fallback accuracy.")
        total, correct = 0, 0
        for p, g in zip(preds, golds):
            for a, b in zip(p, g):
                total += 1
                if a == b:
                    correct += 1
        f1 = correct / total if total > 0 else 0.0
        report = "seqeval not installed, fallback accuracy used"

    return mean_loss, f1, report
