import torch

# Chuẩn hóa dữ liệu về cùng chiều dài
def collate_dl(batch, pad_token_id=0, pad_tag_id=0):
    """
    batch: list of samples from NERDataset.__getitem__
    returns padded tensors: widx, tidx, mask
    """

    batch_size = len(batch)
    max_len = max(len(x["widx"]) for x in batch)

    widx = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    tidx = torch.full((batch_size, max_len), pad_tag_id, dtype=torch.long)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

    words_list = []
    for i, x in enumerate(batch):
        L = len(x['widx'])
        widx[i, :L] = torch.tensor(x['widx'], dtype=torch.long)
        tidx[i, :L] = torch.tensor(x['tidx'], dtype=torch.long)
        mask[i, :L] = True

        words_list.append(x.get("words", []))

    return {
        "widx": widx,
        "tidx": tidx,
        "mask": mask,
        "words": words_list
    }

def collate_phobert(batch, tokenizer):
    
    input_ids = [item["input_ids"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]
    labels = [item["labels"] for item in batch]

    batch_enc = tokenizer.pad(
        {"input_ids": input_ids, "attention_mask": attention_masks},
        padding=True,
        return_tensors="pt"
    )

    max_len = batch_enc["input_ids"].size(1)
    def pad_tensor(seq, pad_value, dtype):
        out = torch.full((max_len,), pad_value, dtype=dtype)
        out[:len(seq)] = seq
        return out

    labels = torch.stack([pad_tensor(l, -100, torch.long) for l in labels])

    return {
        "input_ids": batch_enc["input_ids"],
        "attention_mask": batch_enc["attention_mask"],
        "labels": labels,
    }
