#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Surrogate search for CK Tile kernel configuration optimization.

Uses a trained LGBMRegressor as a cheap surrogate function to search the
discrete kernel parameter space (tile sizes, warp config, pipeline, etc.)
without running actual GPU benchmarks.

Strategies:
  - 'random': Sample N random valid configs, score all, return top-K.
  - 'de': Discrete Differential Evolution with mutation over valid parameter choices.

Usage:
    from search import SurrogateSearch
    from predict import Predictor

    predictor = Predictor("models/gemm_universal_fp8_gfx950")
    searcher = SurrogateSearch(predictor, strategy='random')
    results = searcher.search(
        problem={"m": 128, "n": 1536, "k": 7168, "dtype": "fp8", "layout": "rcr"},
        budget=500,
    )
    # results: [(config_dict, predicted_tflops), ...] sorted descending
"""

import random
from typing import Optional

import numpy as np

from feature_engine import GemmUniversalFeatureEngine
from predict import Predictor


class SurrogateSearch:
    """Search kernel parameter space using ML regressor as surrogate objective.

    Parameters
    ----------
    predictor : Predictor
        Trained predictor with a TFLOPS model.
    feature_engine : GemmUniversalFeatureEngine, optional
        Feature engine for parameter space and validation. If None, uses default.
    strategy : str
        Search strategy: 'random' or 'de' (Discrete Differential Evolution).
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        predictor: Predictor,
        feature_engine: Optional[GemmUniversalFeatureEngine] = None,
        strategy: str = "random",
        seed: int = 42,
    ):
        self._predictor = predictor
        self._fe = feature_engine or GemmUniversalFeatureEngine()
        self._strategy = strategy
        self._rng = random.Random(seed)
        self._np_rng = np.random.RandomState(seed)
        self._param_space = self._fe.get_parameter_space()

    def _sample_random_config(self) -> dict:
        """Sample a single random config from the parameter space."""
        config = {}
        for param, values in self._param_space.items():
            config[param] = self._rng.choice(values)
        return config

    def _sample_valid_config(self, max_attempts: int = 50) -> Optional[dict]:
        """Sample a random config that passes all validation constraints."""
        for _ in range(max_attempts):
            config = self._sample_random_config()
            if self._fe.validate_config(config):
                return config
        return None

    def _score_config(self, problem: dict, config: dict) -> float:
        """Score a config using the predictor."""
        return self._predictor.predict_tflops(problem, config)

    def _search_random(
        self, problem: dict, budget: int, top_k: int
    ) -> list[tuple[dict, float]]:
        """Random search: sample valid configs, score all, return top-K."""
        configs = []
        for _ in range(budget):
            cfg = self._sample_valid_config()
            if cfg is not None:
                configs.append(cfg)

        if not configs:
            return []

        scored = []
        for cfg in configs:
            try:
                score = self._score_config(problem, cfg)
                scored.append((cfg, score))
            except Exception:
                continue

        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    def _search_de(
        self,
        problem: dict,
        budget: int,
        top_k: int,
        pop_size: int = 20,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.7,
    ) -> list[tuple[dict, float]]:
        """Discrete Differential Evolution.

        Uses discrete mutation: randomly swap parameters to other valid values
        from the parameter space (no continuous relaxation + snap).

        Each generation:
          1. For each member of the population, create a trial vector by:
             - Selecting 3 random other members (a, b, c)
             - For each parameter, with probability mutation_rate, take the value
               from a, b, or c (uniform choice among the three donors)
             - With probability crossover_rate, take the trial value; otherwise keep original
          2. Validate the trial; if invalid, resample that parameter from the space
          3. Score the trial; if better than parent, replace
        """
        param_names = list(self._param_space.keys())

        population = []
        for _ in range(pop_size):
            cfg = self._sample_valid_config()
            if cfg is not None:
                score = self._score_config(problem, cfg)
                population.append((cfg, score))

        if len(population) < 4:
            return self._search_random(problem, budget, top_k)

        evals_used = len(population)
        max_gens = (budget - evals_used) // pop_size

        for gen in range(max_gens):
            new_pop = []
            for i, (parent, parent_score) in enumerate(population):
                candidates = [j for j in range(len(population)) if j != i]
                if len(candidates) < 3:
                    new_pop.append((parent, parent_score))
                    continue

                a_idx, b_idx, c_idx = self._rng.sample(candidates, 3)
                a, b, c = (
                    population[a_idx][0],
                    population[b_idx][0],
                    population[c_idx][0],
                )

                trial = dict(parent)
                for param in param_names:
                    if self._rng.random() < mutation_rate:
                        donor = self._rng.choice([a, b, c])
                        trial[param] = donor.get(param, parent.get(param))

                    if self._rng.random() > crossover_rate:
                        trial[param] = parent.get(param)

                if not self._fe.validate_config(trial):
                    for param in param_names:
                        if param in trial and trial[param] not in self._param_space.get(
                            param, [trial[param]]
                        ):
                            trial[param] = self._rng.choice(self._param_space[param])
                    if not self._fe.validate_config(trial):
                        new_pop.append((parent, parent_score))
                        continue

                try:
                    trial_score = self._score_config(problem, trial)
                    evals_used += 1
                except Exception:
                    new_pop.append((parent, parent_score))
                    continue

                if trial_score > parent_score:
                    new_pop.append((trial, trial_score))
                else:
                    new_pop.append((parent, parent_score))

            population = new_pop

        population.sort(key=lambda x: -x[1])
        return population[:top_k]

    def search(
        self,
        problem: dict,
        budget: int = 500,
        top_k: int = 10,
        **kwargs,
    ) -> list[tuple[dict, float]]:
        """Search the kernel parameter space for the best configuration.

        Parameters
        ----------
        problem : dict
            Problem specification: m, n, k, dtype, layout, split_k.
        budget : int
            Maximum number of surrogate evaluations.
        top_k : int
            Number of top configurations to return.
        **kwargs
            Strategy-specific parameters (pop_size, mutation_rate, etc.).

        Returns
        -------
        list of (config_dict, predicted_tflops), sorted descending by TFLOPS.
        """
        if self._strategy == "random":
            return self._search_random(problem, budget, top_k)
        elif self._strategy == "de":
            return self._search_de(problem, budget, top_k, **kwargs)
        else:
            raise ValueError(f"Unknown strategy: {self._strategy}")


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(
        description="Surrogate search for optimal kernel config"
    )
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--dtype", default="fp8")
    parser.add_argument("--layout", default="rcr")
    parser.add_argument("--strategy", default="random", choices=["random", "de"])
    parser.add_argument("--budget", type=int, default=500)
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    predictor = Predictor(args.model_dir)
    searcher = SurrogateSearch(predictor, strategy=args.strategy)
    problem = {
        "m": args.m,
        "n": args.n,
        "k": args.k,
        "dtype": args.dtype,
        "layout": args.layout,
        "split_k": 1,
    }

    print(f"Searching with strategy={args.strategy}, budget={args.budget}...")
    t0 = time.time()
    results = searcher.search(problem, budget=args.budget, top_k=args.top_k)
    elapsed = time.time() - t0

    print(f"\nTop {len(results)} configs found in {elapsed * 1000:.1f}ms:")
    for i, (cfg, tflops) in enumerate(results):
        tile_str = f"{cfg.get('tile_m', '?')}x{cfg.get('tile_n', '?')}x{cfg.get('tile_k', '?')}"
        warp_str = f"{cfg.get('warp_m', '?')}x{cfg.get('warp_n', '?')}x{cfg.get('warp_k', '?')}"
        print(
            f"  #{i + 1}: {tflops:8.2f} TFLOPS  tile={tile_str} warp={warp_str} "
            f"pipeline={cfg.get('pipeline', '?')} scheduler={cfg.get('scheduler', '?')}"
        )
