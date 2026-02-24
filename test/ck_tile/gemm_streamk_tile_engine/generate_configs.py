#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from enum import Enum
from typing import Dict, Tuple, List
import argparse
import json
import os
import sys
from dataclasses import dataclass, field, asdict


@dataclass
class TileConfig:
    """Represents the Tile Config section of a Tile Engine config"""

    tile_m: List[int] = field(default_factory=list)
    tile_n: List[int] = field(default_factory=list)
    tile_k: List[int] = field(default_factory=list)
    warp_m: List[int] = field(default_factory=lambda: [2])
    warp_n: List[int] = field(default_factory=lambda: [2])
    warp_k: List[int] = field(default_factory=lambda: [1])
    warp_tile_m: List[int] = field(default_factory=lambda: [32])
    warp_tile_n: List[int] = field(default_factory=lambda: [32])
    warp_tile_k: List[int] = field(default_factory=lambda: [16])

    def to_dict(self) -> Dict:
        return {k: {"values": v} for k, v in asdict(self).items()}


@dataclass
class TraitConfig:
    """Represents the Trait Config section of a Tile Engine config"""

    pipeline: List[str] = field(default_factory=lambda: ["compv3"])
    epilogue: List[str] = field(default_factory=lambda: ["cshuffle"])
    scheduler: List[str] = field(default_factory=lambda: ["intrawave"])
    pad_m: List[bool] = field(default_factory=lambda: [False])
    pad_n: List[bool] = field(default_factory=lambda: [False])
    pad_k: List[bool] = field(default_factory=lambda: [False])
    persistent: List[bool] = field(default_factory=lambda: [True, False])
    reduction_strategy: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {k: {"values": v} for k, v in asdict(self).items()}


class TestVariant(Enum):
    """Represents a Stream-K test variant"""

    def __init__(
        self,
        val: int,
        reduction_strategy: List[str],
        persistent: List[bool],
        datatypes: List[str],
        description: str,
    ):
        self._value_ = val
        self.reduction_strategy = reduction_strategy
        self.persistent = persistent
        self.datatypes = datatypes
        self.description = description

    ATOMIC_SMOKE = (
        0,
        ["atomic"],
        [True, False],
        ["fp16", "bf16", "fp8", "bf8"],
        "Stream-K atomic smoke tests",
    )
    REDUCTION_SMOKE = (
        2,
        ["linear", "tree"],
        [True, False],
        ["fp16", "bf16", "fp8", "bf8"],
        "Stream-K reduction smoke tests",
    )
    EXTENDED = (
        3,
        ["atomic"],
        [True, False],
        ["fp16", "bf16", "fp8", "bf8"],
        "Stream-K extended smoke tests",
    )

    def apply(self, trait_config: TraitConfig) -> None:
        """Applies the current test variant's persistent and reduction strategy setting to the given trait_config"""
        trait_config.persistent = self.persistent
        trait_config.reduction_strategy = self.reduction_strategy


@dataclass
class ProblemSize:
    """Represents a problem size in a Tile Engine config"""

    m: int
    n: int
    k: int
    variant: TestVariant
    split_k: int = 1

    def to_dict(self) -> Dict:
        return {"m": self.m, "n": self.n, "k": self.k, "split_k": self.split_k}


@dataclass
class Config:
    """Represents a Tile Engine config"""

    description: str
    problem_sizes: list[ProblemSize] = field(default_factory=list)
    tile_config: TileConfig = field(default_factory=TileConfig)
    trait_config: TraitConfig = field(default_factory=TraitConfig)
    k_block_per_cu: int = 1
    permute_n: bool = False

    def add_problem_size(self, problem: ProblemSize) -> None:
        """Adds the given problem to this config's problem_sizes"""
        self.problem_sizes.append(problem)

    def to_dict(self) -> Dict:
        config_dict = {
            "problem": {"description": f"{self.description}"},
            "test_params": {
                "problem_sizes": [ps.to_dict() for ps in self.problem_sizes]
            },
            "tile_config": self.tile_config.to_dict(),
            "trait_config": self.trait_config.to_dict(),
            "k_block_per_cu": self.k_block_per_cu,
            "permute_n": self.permute_n,
        }
        return config_dict

    def write_to_file(self, output_file: str) -> None:
        """Writes this configs to the given output_file in a json format"""
        with open(output_file, "w") as config_file:
            json.dump(self.to_dict(), config_file, indent=4)
            config_file.write("\n")


def create_problem_sizes(
    tile_m: int, tile_n: int, tile_k: int, cu_count: int
) -> List[ProblemSize]:
    """Creates and returns a list of problem sizes using the given arguments"""
    problem_sizes = [
        ProblemSize(256, 256, 256, TestVariant.ATOMIC_SMOKE),
        ProblemSize(tile_m * cu_count, tile_n, tile_k, TestVariant.ATOMIC_SMOKE),
        ProblemSize(
            tile_m * 2, tile_n * 2, cu_count * tile_k, TestVariant.ATOMIC_SMOKE
        ),
        ProblemSize(tile_m, tile_n, cu_count * tile_k, TestVariant.REDUCTION_SMOKE),
        ProblemSize(
            tile_m * 4,
            tile_n,
            tile_k * cu_count + (25 * tile_k),
            TestVariant.REDUCTION_SMOKE,
        ),
        ProblemSize(
            tile_m * 3,
            tile_n * 7,
            tile_k * cu_count + (30 * tile_k),
            TestVariant.REDUCTION_SMOKE,
        ),
        # TODO: Add this test once we determine how to label tests as regresion with tile engine
        # ProblemSize((tile_m * cu_count * 2) + (tile_m * 2), tile_n, 2048, TestVariant.EXTENDED)
    ]

    return problem_sizes


def write_config_files(
    problem_sizes: List[ProblemSize],
    configs_dir_path: str,
    datatype: str,
    tile_sizes: Tuple[int, int, int],
) -> str:
    """Writes the given problem_sizes to a config file and returns the names of the config files written to"""
    config_names = []
    tile_m, tile_n, tile_k = tile_sizes
    tile_config = TileConfig([tile_m], [tile_n], [tile_k])

    # Create a config for each test variant
    for variant in TestVariant:
        problem_sizes_filtered = [ps for ps in problem_sizes if ps.variant == variant]

        if (datatype not in variant.datatypes) or len(problem_sizes_filtered) == 0:
            continue

        trait_config = TraitConfig()
        variant.apply(trait_config)
        config_name = f"streamk_{variant.name.lower()}_tests_config_{datatype}"
        config_names.append(config_name)
        file_path = os.path.join(configs_dir_path, config_name + ".json")
        config = Config(
            variant.description, problem_sizes_filtered, tile_config, trait_config
        )
        config.write_to_file(file_path)

    return config_names


def print_config_names(config_file_names: List[str]) -> None:
    """Prints given config file names as a single semi-colon separated string"""
    print(";".join(config_file_names))


def create_config_files(
    cu_count: int, configs_dir_path: str, tile_sizes: int, datatype: str
) -> None:
    """Creates Stream-K test config files and prints the file names in a semi-colon-separated list"""
    tile_m, tile_n, tile_k = tile_sizes

    problem_sizes = create_problem_sizes(tile_m, tile_n, tile_k, cu_count)
    config_names = write_config_files(
        problem_sizes, configs_dir_path, datatype, tile_sizes
    )
    print_config_names(config_names)


def get_args() -> Tuple[int, str, Tuple[int, int, int], str]:
    """Returns user provided arguments"""

    def tile_sizes_type(val: str):
        sizes = None
        parts = val.split(",")
        if len(parts) != 3:
            raise argparse.ArgumentTypeError(
                "--tiles must contain exactly three comma-separated values (m,n,k), e.g. --tiles 256,256,32"
            )
        try:
            sizes = tuple(int(size) for size in parts)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "--tiles must contain exactly three comma-separated integers (m,n,k), e.g. --tiles 256,256,32"
            )

        return sizes

    parser = argparse.ArgumentParser(description="Create Stream-K test configs")
    parser.add_argument(
        "--cu_count", required=True, help="Number of Compute Units on the device"
    )
    parser.add_argument(
        "--configs_dir_path",
        required=True,
        help="Full path configs directory where config files will be written to",
    )

    parser.add_argument(
        "--tiles",
        required=True,
        type=tile_sizes_type,
        help="Block tile sizes for m, n, and k, respectively. Ex: --tiles 256,256,32",
    )

    parser.add_argument(
        "--datatype",
        choices=["fp16", "bf16", "fp8", "bf8"],
        required=True,
        help="The datatype for which the config is generated.",
    )

    args = parser.parse_args()

    return (int(args.cu_count), args.configs_dir_path, args.tiles, args.datatype)


def main():
    cu_count, configs_dir_path, tile_sizes, datatype = get_args()
    create_config_files(cu_count, configs_dir_path, tile_sizes, datatype)
    sys.exit(0)


if __name__ == "__main__":
    main()
