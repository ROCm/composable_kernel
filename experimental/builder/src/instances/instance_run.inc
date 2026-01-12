
using Builder  = ckb::ConvBuilder<SIGNATURE, ALGORITHM>;
using Instance = Builder::Instance;

auto conv = Instance{};
return std::make_tuple(ckt::run(conv, args, inputs, outputs, s_conf), conv.GetInstanceString());
