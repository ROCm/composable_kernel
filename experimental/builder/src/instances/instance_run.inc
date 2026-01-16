
using Builder  = ckb::ConvBuilder<SIGNATURE, ALGORITHM>;
using Instance = Builder::Instance;

auto conv = Instance{};
bool is_supported;
float avg_time;
std::tie(is_supported, avg_time) = ckt::run(conv, args, inputs, outputs, s_conf);
return std::make_tuple(is_supported, avg_time, conv.GetInstanceString());
