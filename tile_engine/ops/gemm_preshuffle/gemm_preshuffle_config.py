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
    c_datatype: str


@dataclass
class Layout:
    """Configuration class for Layout parameter."""

    a_layout: str
    b_layout: str
    c_layout: str


@dataclass
class ArgumentConfig:
    """Configuration class for Argument parameter."""

    datatypes: DataType
    layouts: Layout

    @classmethod
    def from_args(
        cls: Type["ArgumentConfig"],
        datatype: str,
        layout: str,
    ) -> "ArgumentConfig":
        """configuration loader with validation controls"""

        # [DELETE] TO DO : Make sure whether this validation is accurate or not.

        assert datatype in ["fp16", "bf16", "fp8", "bf8"], (
            f"Invalid datatype string: {datatype} (supported datatypes are [fp16, bf16, fp8, and bf8])"
        )

        a_type = datatype
        b_type = datatype
        c_type = datatype
        if datatype in ["fp8", "bf8"]:
            c_type = "fp16"

        datatypes = DataType(
            a_datatype=a_type,
            b_datatype=b_type,
            c_datatype=c_type,
        )

        layout_parts = layout.lower()
        assert len(layout_parts) == 3, (
            f"Invalid layout string: {layout} (must be 3 characters like 'rcr' where r stands for row major and c stands for column major)"
        )
        assert layout_parts[0] == "r" and layout_parts[1] == "c", (
            f"Invalid matrix_a layout : {layout_parts[0]} or matrix_b layout: {layout_parts[1]} (matrix_a must be 'r' for row major and matrix_b must be 'c' for column major as it is the only supported layout for preshuffle)"
        )
        assert layout_parts[2] == "r", (
            f"Invalid matrix_c layout: {layout_parts[2]} (must be 'r' only as currently we are supporting only row major)"
        )

        layouts = Layout(
            a_layout=layout[0],
            b_layout=layout[1],
            c_layout=layout[2],
        )

        return cls(datatypes=datatypes, layouts=layouts)


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
    persistent: EnumConfigParam


@dataclass
class JsonConfig:
    """Main configuration class for GEMM operations"""

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
                persistent=EnumConfigParam(
                    values=config_dict["trait_config"]["persistent"]["values"]
                ),
            )

            return cls(tile_config=tile_config, trait_config=trait_config)

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {str(e)}")
        except KeyError as e:
            raise KeyError(f"Missing required configuration field: {str(e)}")
