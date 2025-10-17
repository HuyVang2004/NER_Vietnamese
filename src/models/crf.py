import torch
import torch.nn as nn
import math

class CRF(nn.Module):
    """
    CRF layer with log-likelihood loss and Viterbi decode.
    tags: 0..num_tags-1
    We include START and STOP transitions inside matrix.
    """

    def __init__(self, num_tags, pad_idx=0):
        super().__init__()
        self.num_tags = num_tags
        self.pad_idx = pad_idx

        # transitions[i,j] is the score of transitioning FROM i TO j
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        # you can learn start/stop as params too:
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))

    def forward(self, emissions, tags, mask):
        """
        Returns negative log-likelihood (scalar loss)
        emissions: [B, L, num_tags] (raw scores)
        tags: [B, L] gold tag ids
        mask: [B, L] boolean
        """
        log_denominator = self._compute_log_partition_function(emissions, mask)
        log_numerator = self._compute_gold_score(emissions, tags, mask)
        nll = log_denominator - log_numerator
        return nll.mean()

    def _compute_log_partition_function(self, emissions, mask):
        # forward algorithm
        B, L, T = emissions.size()
        mask = mask.float()
        # alpha: [B, T]
        alpha = self.start_transitions + emissions[:, 0]  # [B, T]
        for t in range(1, L):
            emit_t = emissions[:, t].unsqueeze(2)  # [B, T, 1]
            # broadcast: alpha.unsqueeze(2) + transitions (T,T) -> [B,T,T]
            scores = alpha.unsqueeze(2) + self.transitions + emit_t  # [B, T_prev, T_cur]
            # log-sum-exp over previous
            new_alpha = torch.logsumexp(scores, dim=1)  # [B, T_cur]
            # apply mask: if mask=1 take new_alpha else keep old
            mask_t = mask[:, t].unsqueeze(1)
            alpha = new_alpha * mask_t + alpha * (1 - mask_t)
        # end transitions
        alpha = alpha + self.end_transitions
        # log-sum-exp over tags
        return torch.logsumexp(alpha, dim=1)  # [B]

    def _compute_gold_score(self, emissions, tags, mask):
        B, L, T = emissions.size()
        score = self.start_transitions[tags[:, 0]] + emissions[:, 0, :].gather(1, tags[:, 0:1]).squeeze(1)
        for t in range(1, L):
            emit_scores = emissions[:, t, :].gather(1, tags[:, t:t+1]).squeeze(1)  # [B]
            trans_scores = self.transitions[tags[:, t-1], tags[:, t]]  # [B]
            mask_t = mask[:, t].float()
            score = score + (emit_scores + trans_scores) * mask_t
        last_tag_indices = mask.long().sum(dim=1) - 1  # index of last valid tag for each example
        last_tags = tags.gather(1, last_tag_indices.unsqueeze(1)).squeeze(1)
        score = score + self.end_transitions[last_tags]
        return score  # [B]

    def viterbi_decode(self, emissions, mask):
        """
        emissions: [B, L, T]
        mask: [B, L] bool
        returns best_paths: list of list (length per batch)
        """
        B, L, T = emissions.size()
        mask = mask.bool()
        backpointers = []
        # initialize
        v = self.start_transitions + emissions[:, 0]  # [B, T]
        backpointers.append(torch.zeros(B, T, dtype=torch.long, device=emissions.device))
        for t in range(1, L):
            broadcast_v = v.unsqueeze(2)  # [B, T, 1]
            scores = broadcast_v + self.transitions  # [B, T_prev, T_cur]
            best_scores, best_prev_tags = scores.max(dim=1)  # [B, T_cur]
            v = best_scores + emissions[:, t]
            backpointers.append(best_prev_tags)
            # mask: if mask[t]==0 then keep previous v
            mask_t = mask[:, t].unsqueeze(1)
            v = v * mask_t + v * (1 - mask_t)
        # add end transitions
        v = v + self.end_transitions
        best_last_scores, best_last_tags = v.max(dim=1)  # [B]
        # backtrace
        best_paths = []
        for i in range(B):
            seq_len = mask[i].long().sum().item()
            if seq_len == 0:
                best_paths.append([])
                continue
            last_tag = best_last_tags[i].item()
            path = [last_tag]
            for bp_t in reversed(backpointers[1:seq_len]):
                last_tag = bp_t[i, last_tag].item()
                path.append(last_tag)
            path.reverse()
            best_paths.append(path)
        return best_paths
