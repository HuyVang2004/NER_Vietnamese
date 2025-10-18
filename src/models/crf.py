
import torch
import torch.nn as nn
import torch.nn.functional as F


class CRF(nn.Module):
    """CRF layer with numerically stable log-likelihood and Viterbi decoding."""
    def __init__(self, num_tags, pad_idx=None):
        super().__init__()
        self.num_tags = num_tags
        self.pad_idx = pad_idx
        self.START_TAG = num_tags
        self.STOP_TAG = num_tags + 1

        # transition[i, j] = score of transitioning j -> i
        self.transitions = nn.Parameter(torch.empty(num_tags + 2, num_tags + 2))
        nn.init.xavier_uniform_(self.transitions)

        # Disallow transitions into START or out of STOP
        self.transitions.data[:, self.START_TAG] = -1e4
        self.transitions.data[self.STOP_TAG, :] = -1e4

    def _log_sum_exp(self, tensor, dim=-1):
        max_score, _ = tensor.max(dim)
        return max_score + torch.log(torch.sum(torch.exp(tensor - max_score.unsqueeze(dim)), dim))

    def forward(self, emissions, tags, mask):
        """
        emissions: [B, L, num_tags]
        tags: [B, L]
        mask: [B, L]
        """
        log_numerator = self._compute_score(emissions, tags, mask)
        log_denominator = self._compute_log_partition(emissions, mask)
        return torch.mean(log_denominator - log_numerator)

    def _compute_score(self, emissions, tags, mask):
        B, L, N = emissions.size()
        score = torch.zeros(B, device=emissions.device)
        tags = torch.cat([
            torch.full((B, 1), self.START_TAG, dtype=torch.long, device=emissions.device),
            tags
        ], dim=1)

        for t in range(L):
            mask_t = mask[:, t]
            emit_t = emissions[torch.arange(B), t, tags[:, t + 1]]
            trans_t = self.transitions[tags[:, t + 1], tags[:, t]]
            score += (emit_t + trans_t) * mask_t

        last_tags = tags.gather(1, mask.sum(1).long().unsqueeze(1)).squeeze(1)
        score += self.transitions[self.STOP_TAG, last_tags]
        return score

    def _compute_log_partition(self, emissions, mask):
        B, L, N = emissions.size()
        alpha = emissions[:, 0] + self.transitions[:N, self.START_TAG].unsqueeze(0)
        for t in range(1, L):
            emit_t = emissions[:, t].unsqueeze(2)  # [B, N, 1]
            trans = self.transitions[:N, :N].unsqueeze(0)
            alpha_t = alpha.unsqueeze(1) + trans + emit_t
            alpha = self._log_sum_exp(alpha_t, dim=2)
            alpha = torch.where(mask[:, t].unsqueeze(1).bool(), alpha, alpha)
        alpha = alpha + self.transitions[self.STOP_TAG, :N].unsqueeze(0)
        return self._log_sum_exp(alpha, dim=1)

    def decode(self, emissions, mask):
        B, L, N = emissions.size()
        score = emissions[:, 0] + self.transitions[:N, self.START_TAG].unsqueeze(0)
        backpointers = []
        for t in range(1, L):
            next_score = score.unsqueeze(2) + self.transitions[:N, :N].unsqueeze(0)
            best_score, best_tag = next_score.max(1)
            score = best_score + emissions[:, t]
            backpointers.append(best_tag)
        best_score, best_last_tag = score.max(1)
        best_paths = []
        for i in range(B):
            seq_len = mask[i].sum().int().item()
            best_tag = best_last_tag[i].item()
            path = [best_tag]
            for bptr in reversed(backpointers[:seq_len-1]):
                best_tag = bptr[i, best_tag].item()
                path.append(best_tag)
            best_paths.append(path[::-1])
        return best_paths
