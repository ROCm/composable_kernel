import json
from dataclasses import dataclass
from itertools import product


@dataclass
class ParametersBlockwise:
    n: int
    c: int
    h: int
    w: int
    tile_m: int
    tile_n: int
    warp_m: int
    warp_n: int
    thread_tile_m: int
    thread_tile_n: int


class ReduceConfig:
    def __init__(self, config_json_path: str):
        self.config_json_path = config_json_path
        with open(config_json_path, "r") as f:
            self.config_dict = json.load(f)

    def get_parameter_combinations(
        self,
    ):  # TODO: move this method (and the validation one) elsewhere
        n_values = self.config_dict["problem"]["shape"]["n"]["values"]
        c_values = self.config_dict["problem"]["shape"]["c"]["values"]
        h_values = self.config_dict["problem"]["shape"]["h"]["values"]
        w_values = self.config_dict["problem"]["shape"]["w"]["values"]

        tile_m_values = self.config_dict["tile_config"]["tile_m"]["values"]
        tile_n_values = self.config_dict["tile_config"]["tile_n"]["values"]
        warp_m_values = self.config_dict["tile_config"]["warp_tile_m"]["values"]
        warp_n_values = self.config_dict["tile_config"]["warp_tile_n"]["values"]
        thread_tile_m_values = self.config_dict["tile_config"]["thread_tile_m"][
            "values"
        ]
        thread_tile_n_values = self.config_dict["tile_config"]["thread_tile_n"][
            "values"
        ]

        for combo in product(
            n_values,
            c_values,
            h_values,
            w_values,
            tile_m_values,
            tile_n_values,
            warp_m_values,
            warp_n_values,
            thread_tile_m_values,
            thread_tile_n_values,
        ):
            p = ParametersBlockwise(*combo)
            if self.is_valid_combination(p):
                yield p

    def is_valid_combination(self, parameters: ParametersBlockwise) -> bool:
        # Implement your validation logic here
        return True
