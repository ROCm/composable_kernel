# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

# -*- coding: utf-8 -*-

"""
Handles loading, parsing, and validation of JSON configuration parameters.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Union, Tuple, Type, Dict
import json


@dataclass
class EnumConfigParam:
    """Represents an enumeration-type configuration parameter"""

    values: List[Union[int, str, bool]]


@dataclass
class RangeConfigParam:
    """Represents a numeric range-type configuration parameter"""

    min: int
    max: int
    step: int
    exclude: Optional[List[int]]

    def generate_candidates(self) -> List[int]:
        """Generates valid candidates after applying range constraints"""

        if self.min > self.max:
            raise ValueError(f"Invalid range: min({self.min}) > max({self.max})")
        if self.step <= 0:
            raise ValueError(f"Step must be positive, got {self.step}")

        candidates = list(range(self.min, self.max + 1, self.step))

        if hasattr(self, "exclude") and self.exclude:
            if not isinstance(self.exclude, list):
                raise TypeError("exclude must be list type")
            exclude_set = set(self.exclude)
            candidates = [x for x in candidates if x not in exclude_set]

        if not candidates:
            raise ValueError(
                f"No valid candidates for range [{self.min}-{self.max}] "
                f"with step {self.step} and excludes {self.exclude}"
            )

        return candidates


@dataclass
class ProblemConfig:
    """configuration class for problem parameter."""

    datatypes: Tuple[EnumConfigParam, ...]
    layouts: Tuple[EnumConfigParam, ...]

    @property
    def datatype_map(self) -> Dict[str, str]:
        """Get datatype as a key-value map."""
        return {
            "matrix_a": self.datatypes[0].values[0],
            "matrix_b": self.datatypes[1].values[0],
            "matrix_c": self.datatypes[2].values[0],
        }

    @property
    def layout_map(self) -> Dict[str, str]:
        """Get layout as a key-value map."""
        return {
            "matrix_a": self.layouts[0].values[0],
            "matrix_b": self.layouts[1].values[0],
            "matrix_c": self.layouts[2].values[0],
        }


@dataclass
class TileConfig:
    """Configuration class for tile parameter."""

    tile_m: Union[EnumConfigParam, RangeConfigParam]
    tile_n: Union[EnumConfigParam, RangeConfigParam]
    tile_k: Union[EnumConfigParam, RangeConfigParam]

    warp_m: Union[EnumConfigParam, RangeConfigParam]
    warp_n: Union[EnumConfigParam, RangeConfigParam]
    warp_k: Union[EnumConfigParam, RangeConfigParam]

    warp_tile_m: Union[EnumConfigParam, RangeConfigParam]
    warp_tile_n: Union[EnumConfigParam, RangeConfigParam]
    warp_tile_k: Union[EnumConfigParam, RangeConfigParam]


@dataclass
class TraitConfig:
    """Configuration class for kernel traits."""

    pipeline: EnumConfigParam
    scheduler: EnumConfigParam
    epilogue: EnumConfigParam
    pad_m: EnumConfigParam
    pad_n: EnumConfigParam
    pad_k: EnumConfigParam


@dataclass
class GemmConfig:
    """Main configuration class for GEMM operations"""

    problem: ProblemConfig
    tile_config: TileConfig
    trait_config: TraitConfig

    @classmethod
    def from_json(
        cls: Type["GemmConfig"], filepath: str, datatype: str, layout: str
    ) -> "GemmConfig":
        """JSON configuration loader with validation controls"""
        config_path = Path(filepath)

        try:
            if not config_path.exists():
                raise FileNotFoundError(f"Config file {filepath} not found")

            with config_path.open("r") as f:
                config_dict = json.load(f)

            a_type = datatype
            b_type = datatype
            c_type = datatype
            if b_type == "int4":
                a_type = "fp16"
            if b_type in ["bf8", "fp8", "int4"]:
                c_type = "fp16"

            layout_parts = layout.lower()
            assert len(layout_parts) == 3, (
                f"Invalid layout string: {layout} (must be 3 characters like 'rcr' where r stands for row major and c stands for column major)"
            )
            assert layout_parts[0] in ("r", "c"), (
                f"Invalid matrix_a layout: {layout_parts[0]} (must be 'r' for row major or or 'c' for column major)"
            )
            assert layout_parts[1] in ("r", "c"), (
                f"Invalid matrix_a layout: {layout_parts[1]} (must be 'r' for row major or or 'c' for column major)"
            )
            assert layout_parts[2] == "r", (
                f"Invalid matrix_c layout: {layout_parts[2]} (must be 'r' only as currently we are supporting only row major)"
            )
            a_layout = layout_parts[0]
            b_layout = layout_parts[1]
            c_layout = layout_parts[2]

            # Parse problem config
            # TODO: Not reading datatype information from json file.
            problem = ProblemConfig(
                datatypes=(
                    EnumConfigParam(values=[a_type]),
                    EnumConfigParam(values=[b_type]),
                    EnumConfigParam(values=[c_type]),
                ),
                layouts=(
                    EnumConfigParam(values=[a_layout]),
                    EnumConfigParam(values=[b_layout]),
                    EnumConfigParam(values=[c_layout]),
                ),
            )

            # Parse tile config
            def create_param(param_dict):
                if "values" in param_dict:
                    return EnumConfigParam(values=param_dict["values"])
                else:
                    return RangeConfigParam(
                        min=param_dict["min"],
                        max=param_dict["max"],
                        step=param_dict["step"],
                        exclude=param_dict.get("exclude", []),
                    )

            tile_config = TileConfig(
                tile_m=create_param(config_dict["tile_config"]["tile_m"]),
                tile_n=create_param(config_dict["tile_config"]["tile_n"]),
                tile_k=create_param(config_dict["tile_config"]["tile_k"]),
                warp_m=create_param(config_dict["tile_config"]["warp_m"]),
                warp_n=create_param(config_dict["tile_config"]["warp_n"]),
                warp_k=create_param(config_dict["tile_config"]["warp_k"]),
                warp_tile_m=create_param(config_dict["tile_config"]["warp_tile_m"]),
                warp_tile_n=create_param(config_dict["tile_config"]["warp_tile_n"]),
                warp_tile_k=create_param(config_dict["tile_config"]["warp_tile_k"]),
            )

            # Parse trait config
            trait_config = TraitConfig(
                pipeline=EnumConfigParam(
                    values=config_dict["trait_config"]["pipeline"]["values"]
                ),
                scheduler=EnumConfigParam(
                    values=config_dict["trait_config"]["scheduler"]["values"]
                ),
                epilogue=EnumConfigParam(
                    values=config_dict["trait_config"]["epilogue"]["values"]
                ),
                pad_m=EnumConfigParam(
                    values=config_dict["trait_config"]["pad_m"]["values"]
                ),
                pad_n=EnumConfigParam(
                    values=config_dict["trait_config"]["pad_n"]["values"]
                ),
                pad_k=EnumConfigParam(
                    values=config_dict["trait_config"]["pad_k"]["values"]
                ),
            )

            return cls(
                problem=problem, tile_config=tile_config, trait_config=trait_config
            )

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {str(e)}")
        except KeyError as e:
            raise KeyError(f"Missing required configuration field: {str(e)}")
