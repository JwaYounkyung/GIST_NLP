# model utils
import torch


def pad_mask(t, pad):
    mask = (t == pad).unsqueeze(1).unsqueeze(2)
    return mask


def masked_attn_mask(target_len, device=None):
    # Prevent leftward information flow in self-attention.
    ones = torch.ones(target_len, target_len, dtype=torch.uint8,
                      device=device)
    t_self_mask = torch.triu(ones, diagonal=1).bool() # upper triangular part

    return t_self_mask