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

def collate_phobert(batch, pad_tag_id=0):
    input_ids = [item["widx"] for item in batch]
    attention_mask = [item["mask"] for item in batch]
    word_ids_list = [item["word_idx"] for item in batch]
    tag_ids = [item["tidx"] for item in batch]

    max_len = max(len(x) for x in input_ids)

    input_ids_padded, attention_mask_padded, tag_ids_padded = [], [], []

    for i in range(len(batch)):
        word_ids = word_ids_list[i]
        tags = tag_ids[i]
        token_tags = []

        prev_word = None
        for wid in word_ids:
            if wid is None:
                token_tags.append(pad_tag_id)
            elif wid != prev_word:
                token_tags.append(tags[wid])
            else:
                token_tags.append(tags[wid])
            prev_word = wid

        pad_len = max_len - len(token_tags)
        token_tags += [pad_tag_id] * pad_len

        input_ids_padded.append(torch.cat([x for x in input_ids[i]] + [torch.zeros(pad_len, dtype=torch.long)]))
        attention_mask_padded.append(torch.cat([x for x in attention_mask[i]] + [torch.zeros(pad_len, dtype=torch.long)]))
        tag_ids_padded.append(torch.tensor(token_tags, dtype=torch.long))

    return {
        "input_ids": torch.stack(input_ids_padded),
        "attention_mask": torch.stack(attention_mask_padded),
        "labels": torch.stack(tag_ids_padded)
    }

