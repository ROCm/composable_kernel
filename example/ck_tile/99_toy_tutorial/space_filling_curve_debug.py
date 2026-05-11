#!/usr/bin/env python3
"""
Debug Python port of ck_tile::space_filling_curve (space_filling_curve.hpp).
Matches the logic in:
  - access_lengths = tensor_lengths // scalars_per_access (elementwise)
  - ordered_access_lengths = reorder(access_lengths, new2old=dim_access_order)
  - 1D -> multi-index on ordered grid using reverse_exclusive_scan strides
  - forward_sweep + optional snake (SnakeCurved) reversal per axis
  - final: reorder(ordered_sfc, old2new=dim_access_order) * scalars_per_access  (elementwise)

Run:  python3 space_filling_curve_debug.py
Or import SpaceFillingCurve, build like transpose_tile's SFC_Y, and call get_index / get_num_of_access.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import List, Sequence, Tuple, Union

Index = List[int]  # multi_index (Y order of *tensor* lengths, before reorder)


def _reverse_exclusive_scan_multiply(x: List[int], init: int = 1) -> List[int]:
    """CK container_reverse_exclusive_scan(..., multiplies, 1) on a sequence (array case)."""
    n = len(x)
    y = [0] * n
    r = init
    for i in range(n - 1, 0, -1):
        y[i] = r
        r = r * x[i]
    y[0] = r
    return y


def new2old_from_sequence_map(perm: Tuple[int, ...]) -> Tuple[int, ...]:
    """
    CK: sequence_map_inverse. If perm is *new2old* (new_pos -> old_pos), this is a no-op identity check.
    For *old2new* map: old2new[old_i] = new position of old[old_i]; inverse is new2old.
    """
    n = len(perm)
    inv = [0] * n
    for i, p in enumerate(perm):
        inv[p] = i
    return tuple(inv)


def container_reorder_new2old(
    old: Union[Sequence[int], Tuple[int, ...]], new2old: Tuple[int, ...]
) -> Tuple[int, ...]:
    """CK container_reorder_given_new2old: new[i] = old[new2old[i]]. new2old lists for each NEW slot, which OLD index."""
    return tuple(int(old[j]) for j in new2old)


def container_reorder_old2new(
    old: Union[Sequence[int], Tuple[int, ...]], old2new: Tuple[int, ...]
) -> Tuple[int, ...]:
    """CK container_reorder_given_old2new: invert old2new, then new2old."""
    n = len(old2new)
    new2old_ = [0] * n
    for oi in range(n):
        ni = old2new[oi]
        new2old_[ni] = oi
    return container_reorder_new2old(tuple(old), tuple(new2old_))


def get_num_of_access(
    tensor_lengths: Sequence[int], scalars_per_access: Sequence[int]
) -> int:
    assert len(tensor_lengths) == len(scalars_per_access)
    for a, s in zip(tensor_lengths, scalars_per_access):
        assert a % s == 0, f"{a} not divisible by {s}"
    tsize = prod(tensor_lengths)
    svec = prod(scalars_per_access)
    assert tsize % svec == 0
    return tsize // svec


@dataclass
class SpaceFillingCurve:
    """
    template<
      TensorLengths,
      DimAccessOrder,   // sequence used as *new2old* when going from linear access order -> ordered dim layout
      ScalarsPerAccess,
      bool SnakeCurved
    >
    """

    tensor_lengths: Tuple[int, ...]
    # new2old for the *reorder* used on access_lengths: ordered_access_lengths = reorder(lengths, dim_access_order)
    # transpose_tile2d_impl uses identity (0,1,...,n-1).
    dim_access_order: Tuple[int, ...]
    scalars_per_access: Tuple[int, ...]
    snake_curved: bool = True

    def __post_init__(self) -> None:
        assert len(self.tensor_lengths) == len(self.scalars_per_access) == len(self.dim_access_order)
        for a, s in zip(self.tensor_lengths, self.scalars_per_access):
            assert a % s == 0, f"tensor len {a} not divisible by scalars_per_access {s}"

    @property
    def n_dim(self) -> int:
        return len(self.tensor_lengths)

    @property
    def access_lengths(self) -> List[int]:
        return [a // s for a, s in zip(self.tensor_lengths, self.scalars_per_access)]

    @property
    def ordered_access_lengths(self) -> List[int]:
        al = self.access_lengths
        return list(container_reorder_new2old(al, self.dim_access_order))

    @property
    def scalar_per_vector(self) -> int:
        return prod(self.scalars_per_access)

    def get_num_of_access(self) -> int:
        return get_num_of_access(self.tensor_lengths, self.scalars_per_access)

    def _decompose_1d_to_ordered_coords(self, access_idx_1d: int) -> List[int]:
        L = self.ordered_access_lengths
        strides = _reverse_exclusive_scan_multiply(L, 1)
        res = access_idx_1d
        out = []
        for jdim in range(self.n_dim):
            # C++: static_for<0, jdim+1,1> { id = res / stride[k]; res -= id*stride[k] }; return id from last
            d = 0
            for k in range(jdim + 1):
                d = res // strides[k]
                res = res - d * strides[k]
            out.append(int(d))
        return out

    def _forward_sweep(self, ordered_access_idx: Sequence[int]) -> List[bool]:
        n = self.n_dim
        L = self.ordered_access_lengths
        forward = [True] * n
        oa = list(ordered_access_idx)
        for idim in range(1, n):
            tmp = oa[0]
            for j in range(1, idim):
                tmp = tmp * L[j] + oa[j]
            forward[idim] = (tmp % 2) == 0
        return forward

    def get_index(self, access_idx_1d: int) -> Tuple[int, ...]:
        """
        _get_index in C++ returns array (multi_index); get_index wraps in number<> for each.
        Returns the multi-index in *original tensor Y order* (same as CK idx_y_start, then
        you still take .value in C++ for tuple of number).
        """
        oa = self._decompose_1d_to_ordered_coords(access_idx_1d)
        L = self.ordered_access_lengths
        fwd = self._forward_sweep(oa)
        # snake along dimensions
        ordered_sfc: List[int] = []
        for idim in range(self.n_dim):
            v = oa[idim]
            if (not self.snake_curved) or fwd[idim]:
                pass
            else:
                v = L[idim] - 1 - v
            ordered_sfc.append(v)
        # container_reorder_given_old2new(ordered_idx, dim_access_order) * ScalarsPerAccess
        reordered = container_reorder_old2new(tuple(ordered_sfc), self.dim_access_order)
        final = [reordered[i] * int(self.scalars_per_access[i]) for i in range(self.n_dim)]
        return tuple(final)

    def all_indices(self) -> List[Tuple[int, ...]]:
        n = self.get_num_of_access()
        return [self.get_index(i) for i in range(n)]


# --- Same scalars_per_access policy as transpose_tile2d_impl_in_thread (2D) ---


def sfc_scalars_for_transpose_2d(
    y_lengths: Tuple[int, int], vec_length_in: int, n_dim_y: int
) -> Tuple[int, int]:
    """
    y_lengths: (y0, y1), vec_length_in = y_lengths[1] in CK when NDimY=2
    y_dim_vec_in, y_dim_vec_out = 1, 0
    """
    y_dim_vec_in, y_dim_vec_out = 1, 0
    per = [1] * n_dim_y
    if vec_length_in == 1:
        for i in range(n_dim_y):
            per[i] = 1
    else:
        for i in range(n_dim_y):
            per[i] = y_lengths[i] if (i in (y_dim_vec_in, y_dim_vec_out)) else 1
    return (per[0], per[1])


def make_sfc_like_transpose_tile_2d(
    y_lengths: Tuple[int, int], vec_length_in: int, snake: bool = True
) -> SpaceFillingCurve:
    """SFC_Y in transpose_tile2d: DimAccessOrder = 0,1,."""
    n_dim = 2
    sp = sfc_scalars_for_transpose_2d(y_lengths, vec_length_in, n_dim_y=n_dim)
    return SpaceFillingCurve(
        tensor_lengths=tuple(y_lengths),
        dim_access_order=tuple(range(n_dim)),  # identity: sequence<0,1>
        scalars_per_access=sp,
        snake_curved=snake,
    )


def _self_test() -> None:
    # 2D example: y_lengths as in a small tile
    for L0, L1 in [(2, 4), (2, 3), (1, 8)]:
        yl = (L0, L1)
        sfc = make_sfc_like_transpose_tile_2d(yl, vec_length_in=L1, snake=True)
        nacc = sfc.get_num_of_access()
        all_idx = sfc.all_indices()
        assert len(all_idx) == nacc, (yl, nacc, len(all_idx))
        # all indices in box [0, L0)*[0, L1) in scalar Y space, aligned to SFC chunk origin
        for pair in all_idx:
            for a, t in zip(pair, sfc.tensor_lengths):
                assert 0 <= a < t, (pair, sfc.tensor_lengths)
        print(
            f"y_lengths={yl} vec_in={L1} num_access={nacc} scalar_per_vector={sfc.scalar_per_vector}"
        )
        for i, idx in enumerate(all_idx):
            print(f"  access {i:3d} -> start idx (y0,y1) = {idx}")

    # vec_length_in == 1: many more accesses, step 1 in each dim
    yl = (2, 4)
    sfc0 = make_sfc_like_transpose_tile_2d(yl, vec_length_in=1, snake=True)
    assert sfc0.get_num_of_access() == 2 * 4
    print("vec_len_in=1: num_access=8 for 2x4")
    for i in range(8):
        print(f"  {i} -> {sfc0.get_index(i)}")


if __name__ == "__main__":
    _self_test()
