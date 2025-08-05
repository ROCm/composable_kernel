# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

# -*- coding: utf-8 -*-

"""
Handles loading, parsing, and validation of JSON and Argument configuration parameters.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Union, Type
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
class DataType:
    """Configuration class for data type parameter."""

    a_datatype: str
    b_datatype: str
    e_datatype: str
    d0_datatype: str
    d1_datatype: str
    ds_datatype: List[str]


@dataclass
class Layout:
    """Configuration class for Layout parameter."""

    a_layout: str
    b_layout: str
    e_layout: str
    d0_layout: str
    d1_layout: str
    ds_layout: List[str]


@dataclass
class ArgumentConfig:
    """Configuration class for Argument parameter."""

    datatypes: DataType
    layouts: Layout
    function_name: str

    @classmethod
    def from_args(
        cls: Type["ArgumentConfig"],
        datatype: str,
        layout: str,
        elementwise_function: str,
    ) -> "ArgumentConfig":
        """configuration loader with validation controls"""

        datatypes = DataType(
            a_datatype=datatype,
            b_datatype=datatype,
            e_datatype=datatype,
            d0_datatype=datatype,
            d1_datatype=datatype,
            ds_datatype=[datatype, datatype],
        )

        layout_parts = layout.lower()
        assert len(layout_parts) == 4, (
            f"Invalid layout string: {layout} (must be 4 characters like 'rcrr' where r stands for row major and c stands for column major)"
        )
        assert layout_parts[0] in ("r", "c"), (
            f"Invalid matrix_a layout: {layout_parts[0]} (must be 'r' for row major or or 'c' for column major)"
        )
        assert layout_parts[1] in ("r", "c"), (
            f"Invalid matrix_b layout: {layout_parts[1]} (must be 'r' for row major or or 'c' for column major)"
        )
        assert layout_parts[2] == "r", (
            f"Invalid matrix_e layout: {layout_parts[2]} (must be 'r' only as currently we are supporting only row major)"
        )
        assert layout_parts[3] == "r", (
            f"Invalid D dimension layout: {layout_parts[3]} (must be 'r' only as currently we are supporting only row major)"
        )

        layouts = Layout(
            a_layout=layout[0],
            b_layout=layout[1],
            e_layout=layout[2],
            d0_layout=layout[3],
            d1_layout=layout[3],
            ds_layout=[layout[3], layout[3]],
        )
        # Elementwise function name validation
        valid_functions = ["mul", "add", "passthrough"]
        if elementwise_function not in valid_functions:
            raise ValueError(
                f"Invalid elementwise function: {elementwise_function}. "
                f"Valid options are: {', '.join(valid_functions)}"
            )

        # Set the function name based on the elementwise function
        if elementwise_function == "mul":
            function_name = "MultiDMultiply"
        elif elementwise_function == "add":
            function_name = "MultiDAdd"
        elif elementwise_function == "passthrough":
            function_name = "PassThrough"  # TODO Change this

        return cls(datatypes=datatypes, layouts=layouts, function_name=function_name)


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
class JsonConfig:
    """Configuration class for JSON parameter."""

    tile_config: TileConfig
    trait_config: TraitConfig

    @classmethod
    def from_json(cls: Type["JsonConfig"], filepath: str) -> "JsonConfig":
        """JSON configuration loader with validation controls"""
        config_path = Path(filepath)

        try:
            if not config_path.exists():
                raise FileNotFoundError(f"Config file {filepath} not found")

            with config_path.open("r") as f:
                config_dict = json.load(f)

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

            return cls(tile_config=tile_config, trait_config=trait_config)

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {str(e)}")
        except KeyError as e:
            raise KeyError(f"Missing required configuration field: {str(e)}")
