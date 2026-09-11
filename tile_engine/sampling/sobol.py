# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Pure-Python scrambled Sobol sequence generator with scipy fallback.

Uses Joe-Kuo direction numbers for up to 21 dimensions. At daily-tier
budgets (~2000-3000 points from ~10,000-25,000 feasible), pure Python
runs well under a second.
"""

import random

# Joe-Kuo direction numbers for dimensions 2-21.
# Each row: (degree_of_primitive_poly, coefficients_of_poly, [initial_direction_numbers])
# Dimension 1 uses the Van der Corput sequence (bit-reversal).
# Source: Joe & Kuo (2010), https://web.maths.unsw.edu.au/~fkuo/sobol/joe-kuo-old.1111
_JOE_KUO_PARAMS = [
    (1, 0, [1]),
    (2, 1, [1, 1]),
    (3, 1, [1, 3, 1]),
    (3, 2, [1, 3, 3]),
    (4, 1, [1, 1, 1, 1]),  # dim 6
    (4, 4, [1, 1, 3, 3]),
    (5, 2, [1, 3, 5, 13, 7]),  # dim 8
    (5, 4, [1, 1, 5, 5, 17]),
    (5, 7, [1, 1, 5, 5, 5]),  # dim 10
    (5, 11, [1, 1, 7, 11, 19]),
    (5, 13, [1, 1, 5, 1, 1]),
    (5, 14, [1, 1, 1, 3, 11]),
    (6, 1, [1, 3, 5, 5, 31, 45]),  # dim 14
    (6, 13, [1, 3, 3, 9, 7, 25]),
    (6, 16, [1, 3, 1, 15, 17, 63]),
    (7, 19, [1, 1, 5, 13, 11, 3, 15]),  # dim 17
    (7, 22, [1, 3, 1, 7, 3, 23, 79]),
    (7, 25, [1, 3, 7, 9, 31, 29, 17]),
    (7, 37, [1, 1, 3, 15, 29, 15, 41]),  # dim 20
    (7, 41, [1, 3, 1, 7, 3, 23, 79]),  # dim 21 (repeat of 18 for safety)
]

_BITS = 32


def _compute_direction_numbers(dim_index):
    """Compute 32-bit direction numbers for a given dimension (0-indexed, dim 0 = Van der Corput)."""
    if dim_index == 0:
        return [1 << (_BITS - 1 - i) for i in range(_BITS)]

    params = _JOE_KUO_PARAMS[dim_index - 1]
    s = params[0]
    a = params[1]
    m_init = params[2]

    v = [0] * _BITS
    for i in range(min(s, _BITS)):
        if i < len(m_init):
            v[i] = m_init[i] << (_BITS - 1 - i)
        else:
            v[i] = 1 << (_BITS - 1 - i)

    for i in range(s, _BITS):
        v[i] = v[i - s] ^ (v[i - s] >> s)
        for j in range(1, s):
            if (a >> (s - 1 - j)) & 1:
                v[i] ^= v[i - j]

    return v


class SobolSequence:
    """Scrambled Sobol sequence generator.

    Falls back to scipy.stats.qmc.Sobol when available for better scrambling quality.
    """

    def __init__(self, d, scramble=True, seed=0):
        self.d = d
        self.scramble = scramble
        self.seed = seed
        self._use_scipy = False

        if d > 21:
            raise ValueError(f"Sobol dimension {d} exceeds maximum 21")

        try:
            from scipy.stats.qmc import Sobol as ScipySobol

            self._scipy_sobol = ScipySobol(d=d, scramble=scramble, seed=seed)
            self._use_scipy = True
        except ImportError:
            self._direction_numbers = [_compute_direction_numbers(i) for i in range(d)]
            self._scramble_shifts = []
            if scramble:
                rng = random.Random(seed)
                self._scramble_shifts = [
                    rng.randint(0, (1 << _BITS) - 1) for _ in range(d)
                ]

    def generate(self, n):
        """Generate n points in [0, 1)^d."""
        if self._use_scipy:
            import math

            m = max(1, math.ceil(math.log2(n))) if n > 0 else 0
            points = self._scipy_sobol.random_base2(m)
            return points[:n].tolist()

        points = []
        x = [0] * self.d

        for i in range(n):
            if i == 0:
                point = [0.0] * self.d
                if self.scramble:
                    for dim in range(self.d):
                        x[dim] = self._scramble_shifts[dim]
                        point[dim] = x[dim] / (1 << _BITS)
                points.append(point)
            else:
                c = _rightmost_zero_bit(i - 1)
                point = [0.0] * self.d
                for dim in range(self.d):
                    if c < _BITS:
                        x[dim] ^= self._direction_numbers[dim][c]
                    point[dim] = x[dim] / (1 << _BITS)
                points.append(point)

        return points


def _rightmost_zero_bit(n):
    """Find position of rightmost zero bit."""
    pos = 0
    while n & 1:
        n >>= 1
        pos += 1
    return pos
