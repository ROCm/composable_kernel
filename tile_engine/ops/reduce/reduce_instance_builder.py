import argparse
from dataclasses import asdict

from reduce_config import ReduceConfig


from string import Template

# from commons.validation_utils import (
#     is_tile_config_valid,
#     is_trait_combination_valid,
#     get_dtype_string,
#     get_abc_layouts,
# )


class MultiReduceMultiBlockKernelBuilder:
    def __init__(self, working_path, gpu_target, datatype, config_json=None):
        self.working_path = working_path
        self.gpu_target = gpu_target
        self.datatype = datatype

        self.config = ReduceConfig(config_json) if config_json else None

    def _generate_instances(self, template_path: str):
        if not self.config:
            raise ValueError("Configuration not provided.")

        instances = []
        for params in self.config.get_parameter_combinations():
            instance = self._create_instance(params, template_path)
            instances.append(instance)
        return instances  # TODO: write these instance somewhere?

    def _create_instance(self, parameters, template_path: str):
        with open(template_path, "r") as file:
            template_content = file.read()

        template = Template(template_content)
        instance_code = template.substitute(**asdict(parameters))
        return instance_code

    def do_list_blobs(self, template_path: str):
        instances = self._generate_instances(template_path)
        for instance in instances:
            print(instance)  # Or handle the instance code as needed


def main(args):
    variants = {
        "multiblock": {
            "class": MultiReduceMultiBlockKernelBuilder,
            "template": "templates/multi_reduce_multiblock.cpp.template",
        }
    }

    if args.variant and not args.list_blobs:
        raise ValueError("Please provide a Reduction Kernel variant")

    builder = variants.get(args.variant)
    builder_instance = builder["class"](
        working_path=args.working_path,
        gpu_target=args.gpu_target,
        datatype=args.datatype,
        config_json=args.config_json,
    )

    if args.list_blobs:
        builder_instance.do_list_blobs(builder["template"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reduce Instance Builder")

    parser.add_argument(
        "--working_path", type=str, required=True, help="Working directory path"
    )
    parser.add_argument("--datatype", type=str, required=True, help="Data type")
    parser.add_argument(
        "--variant", type=str, required=True, help="Variant: multiblock or threadwise"
    )
    parser.add_argument(
        "--config_json", type=str, required=True, help="Path to config JSON blob"
    )
    parser.add_argument("--list_blobs", action="store_true", help="List blobs")
    parser.add_argument("--gen_blobs", action="store_true", help="Generate blobs")
    parser.add_argument("--gpu_target", type=str, required=True, help="GPU target")

    args = parser.parse_args()

    main(args)

    # m = MultiReduceMultiBlockKernelBuilder("./", "gfx942", "float32", "configs/default_multi_reduce_multiblock_config.json")
    # print(m.config.config_dict)
    # print(list(m.config.get_parameter_combinations()))

    # print(m.generate_instances("templates/test_multi_reduce_multiblock.cpp.template")[0])
