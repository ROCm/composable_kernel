import torch
from typing import Optional

def get_valid_attn_mask(
    device: torch.device,
    causal: bool,
    N: int,
    seq_lengths: torch.Tensor,
    num_targets: Optional[torch.Tensor] = None,
    max_attn_len: int = 0,
    contextual_seq_len: int = 0,
    min_full_attn_seq_len: int = 0,
) -> torch.Tensor:
    ids = torch.arange(0, N, device=device).view(1, N)
    max_ids = seq_lengths.view(-1, 1, 1)
    if contextual_seq_len > 0:
        ids = ids - contextual_seq_len + 1
        ids = torch.clamp(ids, min=0)
        max_ids = max_ids - contextual_seq_len + 1
    if num_targets is not None:
        max_ids = max_ids - num_targets.view(-1, 1, 1)
        ids = torch.clamp(
            ids,
            max=max_ids,
        )
        row_ids = ids.view(-1, N, 1).expand(-1, N, N)
        col_ids = ids.view(-1, 1, N).expand(-1, N, N)
    else:
        row_ids = ids.view(N, 1).expand(N, N)
        col_ids = row_ids.t()
        row_ids = row_ids.view(1, N, N)
        col_ids = col_ids.view(1, N, N)

    row_col_dist = row_ids - col_ids

    ## ensure mask value in diagonal is always 1
    valid_attn_mask = torch.eye(N, device=device, dtype=torch.bool).view(1, N, N)
    if not causal:
        row_col_dist = torch.where(row_col_dist > 0, row_col_dist, -row_col_dist)

    ## 1) for token pair in [seqlen-num_target, N) x [seqlen-num_target, N), row_col_dist is 0
    ## 2) for token pair in [seqlen-num-target, N) x [0, seqlen-num_target), row_col_dist > 0
    ## 3) for token_pair in [0, seq-num_target) x [seqlen-num_target, N). row_col_dist < 0 if causal, else row_col_dist > 0 
    valid_attn_mask = torch.logical_or(valid_attn_mask, row_col_dist > 0)
    if max_attn_len > 0:
        if min_full_attn_seq_len > 0:
            valid_attn_mask = torch.logical_and(
                valid_attn_mask,
                torch.logical_or(
                    row_col_dist <= max_attn_len,
                    row_ids >= max_ids - min_full_attn_seq_len,
                ),
            )
        else:
            valid_attn_mask = torch.logical_and(
                valid_attn_mask, row_col_dist <= max_attn_len
            )
    if contextual_seq_len > 0:
        ## ensure first contextual_seqlen rows (where row_ids==0) attend to all cols less than max_ids
        valid_attn_mask = torch.logical_or(
            valid_attn_mask, torch.logical_and(row_ids == 0, col_ids < max_ids)
        )
    return valid_attn_mask.to(torch.int8)

def main():
    N=64
    contextual_seq_len=3
    max_attn_len=4
    causal=True
    min_full_attn_seq_len=0
    dev_type=torch.device("cpu")
    seq_lengths=torch.tensor((56,60,64), device=dev_type, dtype=torch.int32)
    num_targets=torch.tensor((4,5,6), device=dev_type, dtype=torch.int32)

    valid_attn_mask=get_valid_attn_mask(dev_type,  causal, N, seq_lengths, num_targets, max_attn_len, contextual_seq_len, min_full_attn_seq_len)
    ##torch.set_printoptions(profile="full", linewidth=1024)
    ##print(valid_attn_mask)
    torch.save(valid_attn_mask, "torch_hstu_mask.pt")

if __name__ == "__main__":
    main()


