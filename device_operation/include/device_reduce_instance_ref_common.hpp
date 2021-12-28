#ifndef DEVICE_REDUCE_INSTANCE_REF_COMMON_HPP
#define DEVICE_REDUCE_INSTANCE_REF_COMMON_HPP

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_reduce_instance {

#define ADD_INST_REF_BY_TYPE(key,inT,compT,outT,reduceOp,nanOpt,indicesOpt,rank,...) \
   extern template void add_device_reduce_instance_##key<inT,compT,outT,rank,Sequence<__VA_ARGS__>,reduceOp,nanOpt,indicesOpt>( \
                        std::vector<DeviceReducePtr<inT,compT,outT,rank,Sequence<__VA_ARGS__>,reduceOp,nanOpt,indicesOpt>> & device_op_instances)

#define ADD_INST_REF_BY_ID(key,inT,compT,outT,reduceOp,nanOpt,indicesOpt,rank,...) \
        ADD_INST_REF_BY_TYPE(key,inT,compT,outT,static_cast<ReduceTensorOp_t>(reduceOp),static_cast<NanPropagation_t>(nanOpt),static_cast<ReduceTensorIndices_t>(indicesOpt),rank,__VA_ARGS__) 
    
    
} // namespace device_reduce_instance
} // namespace device
} // namespace tensor_operation

} // namespace ck

#endif


