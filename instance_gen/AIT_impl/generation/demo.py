import jinja2

SHAPE_EVAL_TEMPLATE = jinja2.Template(
    """
    int M = *in_{{ range(rank - 1)|join(' * *in_') }};
    int N = *in_{{rank - 1}};
    static constexpr auto I0 = Number<0>{};
            static constexpr auto I1 = Number<1>{};
            static constexpr auto I2 = Number<2>{};
            static constexpr auto I3 = Number<3>{};
            static constexpr auto I4 = Number<4>{};
            static constexpr auto I5 = Number<5>{};

            static constexpr auto K1Number = Number<K1>{};
    """
)

output = SHAPE_EVAL_TEMPLATE.render(rank=2);
print (output)