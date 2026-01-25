import math
import torch
from typing import Optional

def get_valid_attn_mask_v2(
    device: torch.device,
    causal: bool,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    seq_lengths_q: torch.Tensor,
    seq_lengths_kv: torch.Tensor,
    num_targets: Optional[torch.Tensor] = None,
    max_attn_len: int = 0,
    contextual_seq_len: int = 0,
    min_full_attn_seq_len: int = 0,
) -> torch.Tensor:
    N = max(max_seqlen_q, max_seqlen_kv)
    ids = torch.arange(0, N, device=device).view(1, N)
    max_ids_q = seq_lengths_q.view(-1, 1, 1)
    max_ids_kv = seq_lengths_kv.view(-1, 1, 1)
    diff_q_kv = max_ids_kv - max_ids_q
    if contextual_seq_len > 0:
        ids = ids - contextual_seq_len + 1
        ids = torch.clamp(ids, min=0)
        max_ids_q = max_ids_q - contextual_seq_len + 1
        max_ids_kv = max_ids_kv - contextual_seq_len + 1
    if num_targets is not None:
        max_ids_q = max_ids_q - num_targets.view(-1, 1, 1)
        max_ids_kv = max_ids_kv - num_targets.view(-1, 1, 1)

    raw_row_ids = torch.clamp(
        ids,
        max=max_ids_q,
    )
    raw_row_ids = raw_row_ids + diff_q_kv
    max_ids_q = max_ids_q + diff_q_kv
    raw_col_ids = torch.clamp(
        ids,
        max=max_ids_kv,
    )
    row_ids = raw_row_ids.view(-1, N, 1).expand(-1, N, N)
    col_ids = raw_col_ids.view(-1, 1, N).expand(-1, N, N)

    row_col_dist = row_ids - col_ids

    ## ensure mask value in diagonal is always 1
    ##valid_attn_mask = torch.eye(N, device=device, dtype=torch.bool).view(1, N, N)
    valid_attn_mask = torch.zeros_like(row_col_dist).to(torch.bool)
    for idx0 in range(valid_attn_mask.size(0)):
        for idx1 in torch.arange(max_seqlen_q):
                valid_attn_mask[idx0, idx1, idx1 + diff_q_kv[idx0]] = 1  

    if not causal:
        row_col_dist = torch.where(row_col_dist > 0, row_col_dist, -row_col_dist)

    ## 1) for token pair in [seqlen-num_target, N) x [seqlen-num_target, N), row_col_dist is 0
    ## 2) for token pair in [seqlen-num-target, N) x [0, seqlen-num_target), row_col_dist > 0
    ## 3) for token_pair in [0, seqlen-num_target) x [seqlen-num_target, N). row_col_dist < 0 if causal, else row_col_dist > 0
    valid_attn_mask = torch.logical_or(valid_attn_mask, row_col_dist > 0)
    if max_attn_len > 0:
        if min_full_attn_seq_len > 0:
            valid_attn_mask = torch.logical_and(
                valid_attn_mask,
                torch.logical_or(
                    row_col_dist <= max_attn_len,
                    row_ids >= max_ids_q - min_full_attn_seq_len,
                ),
            )
        else:
            valid_attn_mask = torch.logical_and(
                valid_attn_mask, row_col_dist <= max_attn_len
            )
    if contextual_seq_len > 0:
        ## ensure first contextual_seqlen rows (where row_ids==0) attend to all cols less than max_ids
        valid_attn_mask = torch.logical_or(
            valid_attn_mask, torch.logical_and(row_ids == diff_q_kv, col_ids < max_ids_kv)
        )

    fit_valid_attn_mask = valid_attn_mask[:, :max_seqlen_q, :]

    return fit_valid_attn_mask.to(torch.int8)

def main():
    max_seqlen_q=64
    max_seqlen_kv=80
    contextual_seq_len=3
    max_attn_len=0
    causal=True
    min_full_attn_seq_len=0
    dev_type=torch.device("cpu")
    seq_lengths_q=torch.tensor((56,60,64), device=dev_type, dtype=torch.int32)
    seq_lengths_kv=torch.tensor((70,76,80), device=dev_type, dtype=torch.int32)
    num_targets=torch.tensor((4,5,6), device=dev_type, dtype=torch.int32)

    valid_attn_mask=get_valid_attn_mask_v2(dev_type,  causal, max_seqlen_q, max_seqlen_kv, seq_lengths_q, seq_lengths_kv, num_targets, max_attn_len, contextual_seq_len, min_full_attn_seq_len)
    torch.save(valid_attn_mask, "torch_hstu_mask_0.pt")

    max_attn_len=4
    min_full_attn_seq_len=6
    valid_attn_mask=get_valid_attn_mask_v2(dev_type,  causal, max_seqlen_q, max_seqlen_kv, seq_lengths_q, seq_lengths_kv, num_targets, max_attn_len, contextual_seq_len, min_full_attn_seq_len)
    torch.save(valid_attn_mask, "torch_hstu_mask_1.pt")

if __name__ == "__main__":
    main()


