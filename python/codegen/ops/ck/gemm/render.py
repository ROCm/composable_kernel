from .instance import GEMM
from string import Template

instance_template = Template(r"""
namespace $instance_name {
    using Instance = $ck_class<$template_params>;
}
""")

def render(instance: GEMM):
    template_params = []
    for field_name, field_value in instance.dict_items():
        if isinstance(field_value, tuple):
            tuple_elements = ", ".join(map(str, iter(field_value)))
            if "ds" in field_name:  # element type and layout for bias
                arg = f"/* {field_name} */ Tuple<{tuple_elements}>"
            else:  # tile shape
                arg = f"/* {field_name} */ S<{tuple_elements}>"
            template_params.append(arg)
        else:
            if field_value is not None:
                template_params.append(f"/* {field_name} */ {field_value}")
    template_params_separator = ",\n" + 12 * " "
    return instance_template.substitute(
        instance_name=instance.name(),
        ck_class="DeviceGemmMultiD_Xdl_CShuffle_V3",
        template_params=template_params_separator.join(template_params)
    )
