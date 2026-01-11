
using Builder  = ckb::ConvBuilder<SIGNATURE, ALGORITHM>;
using Instance = Builder::Instance;

auto conv = Instance{};
return std::make_tuple(ckt::run(conv, args, inputs, outputs, 1 /*k_batch*/, s_conf),
                       conv.GetInstanceString());
