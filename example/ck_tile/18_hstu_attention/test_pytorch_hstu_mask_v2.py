import math
from typing import Optional

import torch


def get_valid_attn_mask_v2(
    device: torch.device,
    causal: bool,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    seq_lengths_q: torch.Tensor,
    seq_lengths_kv: torch.Tensor,
    num_targets: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Generate attention mask for HSTU attention.

    This implementation matches the Hammer/PyTorch reference (_get_valid_attn_mask_v2)
    with targets_in_kv=False, meaning:
    - Q sequence: [UIH_Q][Targets] with length seq_lengths_q
    - KV sequence: [UIH_KV] only with length seq_lengths_kv (NO targets in KV)

    For causal attention with num_targets:
    - UIH rows (col_ids < uih_lengths_q): apply causal mask (shifted_col_ids >= row_ids)
    - Target rows (col_ids >= uih_lengths_q): can attend to everything in KV (full attention)
    """
    # Create position indices - matching Hammer convention:
    # col_ids indexes Q dimension (rows in attention matrix when viewed as Q x KV)
    # row_ids indexes KV dimension (columns in attention matrix)
    col_ids = torch.arange(0, max_seqlen_q, device=device).view(1, max_seqlen_q, 1)
    row_ids = torch.arange(0, max_seqlen_kv, device=device).view(1, 1, max_seqlen_kv)

    # Boundary mask: positions within valid sequence bounds
    in_boundary_valid_attn_mask = torch.logical_and(
        row_ids < seq_lengths_kv.view(-1, 1, 1), col_ids < seq_lengths_q.view(-1, 1, 1)
    )

    if causal:
        if num_targets is None:
            # Causal without num_targets: simple shifted causal mask
            delta_col_ids = seq_lengths_kv - seq_lengths_q
            shifted_col_ids = col_ids + delta_col_ids.view(-1, 1, 1)
            causal_mask = shifted_col_ids >= row_ids
            return torch.logical_and(in_boundary_valid_attn_mask, causal_mask).to(
                torch.int8
            )
        else:
            # Causal with num_targets and targets_in_kv=False
            # This exactly mirrors the Hammer logic with targets_in_kv=False
            uih_lengths_q = seq_lengths_q - num_targets
            delta_col_ids = seq_lengths_kv - uih_lengths_q
            # targets_in_kv=False: NO subtraction of num_targets from delta_col_ids
            shifted_col_ids = col_ids + delta_col_ids.view(-1, 1, 1)

            # UIH rows: apply causal mask
            causal_mask = torch.logical_and(
                col_ids < uih_lengths_q.view(-1, 1, 1), shifted_col_ids >= row_ids
            )

            # Target rows: full attention to KV (no additional constraint for targets_in_kv=False)
            target_mask = col_ids >= uih_lengths_q.view(-1, 1, 1)

            return torch.logical_and(
                in_boundary_valid_attn_mask, torch.logical_or(causal_mask, target_mask)
            ).to(torch.int8)
    else:
        # Non-causal: everything in bounds is allowed
        return in_boundary_valid_attn_mask.to(torch.int8)


def main():
    max_seqlen_q = 64
    max_seqlen_kv = 80
    causal = True
    dev_type = torch.device("cpu")
    seq_lengths_q = torch.tensor((56, 60, 64), device=dev_type, dtype=torch.int32)
    seq_lengths_kv = torch.tensor((70, 76, 80), device=dev_type, dtype=torch.int32)
    num_targets = torch.tensor((4, 5, 6), device=dev_type, dtype=torch.int32)

    valid_attn_mask = get_valid_attn_mask_v2(
        dev_type,
        causal,
        max_seqlen_q,
        max_seqlen_kv,
        seq_lengths_q,
        seq_lengths_kv,
        num_targets,
    )
    torch.save(valid_attn_mask, "torch_hstu_mask_0.pt")

if __name__ == "__main__":
    main()
