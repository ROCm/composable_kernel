
using Builder  = ckb::ConvBuilder<SIGNATURE, ALGORITHM>;
using Instance = Builder::Instance;

auto conv             = Instance{};
ckt::RunResult result = ckt::run(conv, args, inputs, outputs, s_conf);
return std::make_tuple(result.is_supported(), result.runtime, conv.GetInstanceString());
