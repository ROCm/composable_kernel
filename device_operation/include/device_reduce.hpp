#ifndef DEVICE_REDUCE_HPP
#define DEVICE_REDUCE_HPP

#include <vector>
#include <memory>

#include "common_header.hpp"
#include "device_base.hpp"
#include "reduction_enums.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename inType, typename compType, typename outType, int rank, typename reduceDims, 
	  ReduceTensorOp_t reduceOp, NanPropagation_t nanOpt, ReduceTensorIndices_t indicesOpt>
struct DeviceReduce : public BaseOperator
{
    virtual size_t getWorkspaceSize(const std::vector<int> & inLengths) = 0; 

    virtual bool hasFurtherCall() { return(false); };  

    virtual std::vector<int> getWorkspace2dLengths(const BaseArgument *argPtr) 
    { 
        (void)argPtr; 	
        return( std::vector<int>{0, 0} ); 
    }; 

    virtual std::unique_ptr<BaseArgument> MakeArgumentPointer(const std::vector<int> & inLengths, const std::vector<int> & inStrides,
		                                              const std::vector<int> & outLengths, const std::vector<int> & outStrides, 
		                                              float alpha, float beta, const void *in_dev, void *out_dev, void *out_indices_dev, void *workspace_dev) = 0; 

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

template <typename inType, typename compType, typename outType, int rank, typename reduceDims, 
	  ReduceTensorOp_t reduceOp, NanPropagation_t nanOpt, ReduceTensorIndices_t indicesOpt>
using DeviceReducePtr = std::unique_ptr< DeviceReduce<inType, compType, outType, rank, reduceDims, reduceOp, nanOpt, indicesOpt> >;

} // namespace device
} // namespace tensor_operation
} // namespace ck
#endif
