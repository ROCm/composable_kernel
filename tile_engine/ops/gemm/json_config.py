# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

# -*- coding: utf-8 -*-

"""
Handles loading, parsing, and validation of JSON configuration parameters.
"""

from pathlib import Path
from pydantic import BaseModel, model_validator, field_validator, ValidationInfo, Field, ValidationError
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union, Tuple, Type
import json


class BaseConfigParam(BaseModel):
    """Base model for configuration parameters, enforcing mode validation."""

    @model_validator(mode='before')
    def validate_mode_exclusivity(cls, data: Dict) -> Dict:
        mode_requirements = {
            'enum': {'required': ['values'], 'optional': []},
            'range': {'required': ['min', 'max'], 'optional': ['step']}
        }

        active_modes = []
        for mode, reqs in mode_requirements.items():
            required_fields = reqs['required']
            if all(field in data for field in required_fields):
                active_modes.append(mode)

        if len(active_modes) > 1:
            raise ValidationError(
                f"Configuration conflict: Multiple active modes detected {active_modes}"
            )

        if not active_modes:
            raise ValidationError(
                "No valid configuration mode detected. Must provide either: "
                "- enum: 'values' list\n"
                "- range: 'min'/'max' with optional 'step'"
            )

        return data


class EnumConfigParam(BaseConfigParam):
    """Represents an enumeration-type configuration parameter"""
    values: List[Union[int, str, bool]] = Field(
        ...,
        min_items=1,
        description="Allowed values for enum selection"
    )

    @field_validator("values")
    def validate_enum_values(cls, v, info: ValidationInfo) -> Any:
        # Type validation
        valid_types = (int, str, bool)
        for idx, item in enumerate(v):
            if not isinstance(item, valid_types):
                raise ValidationError(
                    f"Invalid type '{type(item).__name__}' at index {idx}. "
                    f"Allowed types: {[t.__name__ for t in valid_types]}",
                    [{
                        'type': 'invalid_type',
                        'ctx': {
                            'position': idx,
                            'invalid_type': type(item).__name__,
                            'allowed_types': [t.__name__ for t in valid_types]
                        }
                    }]
                )

            # String content validation
            if isinstance(item, str) and not item.strip():
                raise ValidationError(
                    "Empty string not allowed in enum values",
                    [{
                        'type': 'empty_string',
                        'ctx': {'position': idx}
                    }]
                )

        # Duplicate check
        unique_values = set()
        for idx, item in enumerate(v):
            if item in unique_values:
                raise ValidationError(
                    f"Duplicate value '{item}' at index {idx}",
                    [{
                        'type': 'duplicate_value',
                        'ctx': {'position': idx, 'value': item}
                    }]
                )
            unique_values.add(item)

        return v


class RangeConfigParam(BaseConfigParam):
    """Represents a numeric range-type configuration parameter"""
    min: int = Field(
        ...,
        description="Lower boundary for range mode"
    )

    max: int = Field(
        ...,
        description="Upper boundary for range mode"
    )

    step: int = Field(
        default=1,
        ge=1,
        description="Increment step between values (minimum 1)"
    )

    exclude: Optional[List[int]] = Field(
        default=None,
        description="Values to exclude from the range (must be within [min, max])"
    )

    @model_validator(mode='before')
    def validate_min_max_relationship(cls, data: dict) -> dict:
        """Validates range boundaries and step compatibility"""
        min_val = data.get('min')
        max_val = data.get('max')
        if min_val is not None and max_val is not None and min_val > max_val:
            raise ValueError("min: {min_val} must be less than max: {max_val}")
        # Pre-validate candidate generation to catch empty ranges
        if all(key in data for key in ('min', 'max', 'step')):
            try:
                candidates = list(
                    range(
                        data['min'],
                        data['max'] + 1,
                        data['step']))
                if not candidates:
                    raise ValueError("Empty candidate list with current step")
            except ValueError as e:
                raise ValueError(f"Invalid step configuration: {str(e)}")

        return data

    @field_validator('step')
    def validate_step_value(cls, v: int) -> int:
        """Ensures step is a valid positive integer"""
        if v <= 0:
            raise ValueError(f"Step: {v} must be a positive integer")
        return v

    @field_validator('exclude')
    def validate_exclusion_range(cls, v: list, values: ValidationInfo) -> list:
        """Validates exclusion list against range constraints"""
        if not v:
            return v

        data = values.data
        if 'min' not in data or 'max' not in data:
            raise ValueError("Missing min/max for exclusion validation")

        min_val = data['min']
        max_val = data['max']
        step_val = data.get('step', 1)

        # Check for duplicate exclusions
        if len(v) != len(set(v)):
            raise ValueError("Exclude list contains duplicate values")

        # Validate value boundaries
        out_of_bounds = [x for x in v if not (min_val <= x <= max_val)]
        if out_of_bounds:
            raise ValueError(f"Excluded values {out_of_bounds} out of bounds")

        # Verify step alignment
        misaligned = [x for x in v if (x - min_val) % step_val != 0]
        if misaligned:
            raise ValueError(
                f"Misaligned exclude values {misaligned} with step {step_val}")

        # Detect non-existent candidates in exclusion list
        try:
            candidates = list(range(min_val, max_val + 1, step_val))
            ghost_excludes = [x for x in v if x not in candidates]
            if ghost_excludes:
                raise ValueError(
                    f"Excludes {ghost_excludes} not in candidate list")
        except ValueError as e:
            raise ValueError(f"Invalid configuration: {str(e)}")

        return v

    def generate_candidates(self) -> List[int]:
        """Generates valid candidates after applying range constraints"""
        candidates = list(range(self.min, self.max + 1, self.step))

        if self.exclude:
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
    datatypes: Tuple[EnumConfigParam, ...] = Field(
        default_factory=lambda: (
            EnumConfigParam(values=["fp16"]),
            EnumConfigParam(values=["fp16"]),
            EnumConfigParam(values=["fp16"])
        )
    )

    layouts: Tuple[EnumConfigParam, ...] = Field(
        default_factory=lambda: (
            EnumConfigParam(values=["r"]),
            EnumConfigParam(values=["c"]),
            EnumConfigParam(values=["r"])
        )
    )

    @property
    def datatype_map(self) -> dict[str, str]:
        """Get current layout selections as a key-value map."""
        return {
            'matrix_a': self.datatypes[0].values[0],
            'matrix_b': self.datatypes[1].values[0],
            'matrix_c': self.datatypes[2].values[0]
        }

    @property
    def layout_map(self) -> dict[str, str]:
        """Get current layout selections as a key-value map."""
        return {
            'matrix_a': self.layouts[0].values[0],
            'matrix_b': self.layouts[1].values[0],
            'matrix_c': self.layouts[2].values[0]
        }


@dataclass
class TileConfig:
    """configuration class for tile parameter."""
    tile_m: Union[EnumConfigParam, RangeConfigParam] = Field(
        default_factory=lambda: EnumConfigParam(
            values=[256]
        )
    )
    tile_n: Union[EnumConfigParam, RangeConfigParam] = Field(
        default_factory=lambda: EnumConfigParam(
            values=[256]
        )
    )
    tile_k: Union[EnumConfigParam, RangeConfigParam] = Field(
        default_factory=lambda: EnumConfigParam(
            values=[256]
        )
    )

    warp_m: Union[EnumConfigParam, RangeConfigParam] = Field(
        default_factory=lambda: EnumConfigParam(
            values=[8]
        )
    )
    warp_n: Union[EnumConfigParam, RangeConfigParam] = Field(
        default_factory=lambda: EnumConfigParam(
            values=[8]
        )
    )
    warp_k: Union[EnumConfigParam, RangeConfigParam] = Field(
        default_factory=lambda: EnumConfigParam(
            values=[8]
        )
    )

    warp_tile_m: Union[EnumConfigParam, RangeConfigParam] = Field(
        default_factory=lambda: EnumConfigParam(
            values=[8]
        )
    )
    warp_tile_n: Union[EnumConfigParam, RangeConfigParam] = Field(
        default_factory=lambda: EnumConfigParam(
            values=[8]
        )
    )
    warp_tile_k: Union[EnumConfigParam, RangeConfigParam] = Field(
        default_factory=lambda: EnumConfigParam(
            values=[8]
        )
    )


@dataclass
class TraitConfig:
    """configuration class for kernel traits."""
    pipeline: EnumConfigParam = Field(
        default_factory=lambda: EnumConfigParam(values=['compv3']))

    scheduler: EnumConfigParam = Field(
        default_factory=lambda: EnumConfigParam(values=['intrawave'])
    )

    epilogue: EnumConfigParam = Field(
        default_factory=lambda: EnumConfigParam(values=['default'])
    )

    pad_m: EnumConfigParam = Field(
        default_factory=lambda: EnumConfigParam(values=[False])
    )

    pad_n: EnumConfigParam = Field(
        default_factory=lambda: EnumConfigParam(values=[False])
    )

    pad_k: EnumConfigParam = Field(
        default_factory=lambda: EnumConfigParam(values=[False])
    )


class GemmConfig(BaseModel):
    """Main configuration class for GEMM operations """
    problem: ProblemConfig
    tile_config: TileConfig
    trait_config: TraitConfig

    @classmethod
    def from_json(cls: Type["GemmConfig"], filepath: str,
                  validate_nested: bool = True) -> "GemmConfig":
        """JSON configuration loader with validation controls"""

        config_path = Path(filepath)

        try:
            if not config_path.exists():
                raise FileNotFoundError(f"Config file {filepath} not found")
            config_path.stat()

            with open(filepath, 'r') as f:
                try:
                    config_dict = json.load(f)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"JSON parsing failed in {filepath}\n"
                        f"Error at line {e.lineno}: {e.msg}"
                    ) from e

            if validate_nested:
                return cls.model_validate(
                    config_dict,
                    context={'validating': True}
                )
            else:
                required_fields = {'problem', 'tile_config', 'trait_config'}
                if missing := required_fields - config_dict.keys():
                    raise ValueError(
                        f"Missing required fields: {missing}"
                    )
                return cls.model_construct(**config_dict)

        except ValidationError as ve:
            error_msgs = [
                f"[{'->'.join(map(str, err['loc']))}] "
                f"{err['msg']} (received: {err['input']!r})"
                for err in ve.errors()
            ]
            raise ValueError(
                "Configuration validation failed:\n" + "\n".join(error_msgs)
            ) from ve

        except PermissionError as pe:
            raise RuntimeError(
                f"Permission denied accessing {filepath}"
            )
