
using Builder      = ckb::ConvBuilder<SIGNATURE, ALGORITHM>;
using ConvInstance = Builder::Instance;

auto conv = ConvInstance{};

auto result = [&]<auto Sig, auto Alg>() {
    if constexpr(ConvDirectionIsBackwardWeight<Sig>)
    {
        if constexpr(ckb::SpecifiesTileOptimizations<decltype(Alg)> && Alg.optimizations.two_stage)
        {
            using ElementwiseOpBuilder  = ckf::ElementwiseOpTileFactory<Sig, Alg>;
            using ElementwiseOpInstance = ElementwiseOpBuilder::Instance;
            auto elementwise_op         = ElementwiseOpInstance{};
            return ckt::run(conv, elementwise_op, args, inputs, outputs, s_conf);
        }
        else
        {
            return ckt::run(conv, args, inputs, outputs, s_conf);
        }
    }
    else
    {
        return ckt::run(conv, args, inputs, outputs, s_conf);
    }
}.template operator()<SIGNATURE, ALGORITHM>();

return std::make_tuple(result.is_supported(), result.runtime, conv.GetInstanceString());
