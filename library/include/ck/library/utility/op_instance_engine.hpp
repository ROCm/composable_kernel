#pragma once

#include <memory>
#include <tuple>
#include <utility>

#include "check_err.hpp"
#include "device_base.hpp"
#include "functional2.hpp"

namespace ck {
namespace utils {

struct ProfileBestConfig
{
    std::string best_op_name;
    float best_avg_time   = std::numeric_limits<float>::max();
    float best_tflops     = std::numeric_limits<float>::max();
    float best_gb_per_sec = std::numeric_limits<float>::max();
};

/**
 * @brief      This class describes an operation instance(s).
 *
 *             Op instance defines a particular specializations of operator
 *             template. Thanks to this specific input/output data types, data
 *             layouts and modifying elementwise operations it is able to create
 *             it's input/output tensors, provide pointers to instances which
 *             can execute it and all operation specific parameters.
 */
template <typename OutDataType, typename... InArgTypes>
class OpInstance
{
    public:
    template <typename T>
    using TensorPtr      = std::unique_ptr<Tensor<T>>;
    using InTensorsTuple = std::tuple<TensorPtr<InArgTypes>...>;
    using DeviceMemPtr   = std::unique_ptr<DeviceMem>;
    using DeviceBuffers  = std::vector<DeviceMemPtr>;

    OpInstance()                  = default;
    OpInstance(const OpInstance&) = default;
    OpInstance& operator=(const OpInstance&) = default;
    virtual ~OpInstance(){};

    virtual InTensorsTuple GetInputTensors() const         = 0;
    virtual TensorPtr<OutDataType> GetOutputTensor() const = 0;
    virtual std::unique_ptr<tensor_operation::device::BaseInvoker>
    MakeInvokerPointer(tensor_operation::device::BaseOperator*) const = 0;
    virtual std::unique_ptr<tensor_operation::device::BaseArgument>
    MakeArgumentPointer(tensor_operation::device::BaseOperator*,
                        const DeviceBuffers&,
                        const DeviceMemPtr&) const = 0;
    virtual std::size_t getFlops() const           = 0;
    virtual std::size_t getBtype() const           = 0;
};

/**
 * @brief      A generic operation instance run engine.
 */
template <typename OutDataType, typename... InArgTypes>
class OpInstanceRunEngine
{
    public:
    using OpInstanceT = OpInstance<InArgTypes..., OutDataType>;
    template <typename T>
    using TensorPtr      = std::unique_ptr<Tensor<T>>;
    using DeviceMemPtr   = std::unique_ptr<DeviceMem>;
    using InTensorsTuple = std::tuple<TensorPtr<InArgTypes>...>;
    using DeviceBuffers  = std::vector<DeviceMemPtr>;

    OpInstanceRunEngine() = delete;

    template <typename ReferenceOp>
    OpInstanceRunEngine(const OpInstanceT& op_instance, const ReferenceOp& reference_op)
        : op_instance_{op_instance}
    {
        in_tensors_ = op_instance_.GetInputTensors();
        out_tensor_ = op_instance_.GetOutputTensor();
        ref_output_ = op_instance_.GetOutputTensor();

        constexpr std::size_t N_IN_ARGS = std::tuple_size_v<InTensorsTuple>;
        CallRefOpUnpackArgs(reference_op, std::make_index_sequence<N_IN_ARGS>{});

        ck::static_for<0, N_IN_ARGS, 1>{}([this](auto i) {
            // This is ugly... should somehow iterate also through InArgTypes to get specific
            // data type
            const auto& ts = std::get<i>(in_tensors_);
            using vec_type = decltype(ts->mData);
            this->in_device_buffers_
                .emplace_back(std::make_unique<DeviceMem>(sizeof(typename vec_type::value_type) *
                                                          ts->mDesc.GetElementSpace()))
                ->ToDevice(ts->mData.data());
        });

        using vec_type     = decltype(out_tensor_->mData);
        out_device_buffer_ = std::make_unique<DeviceMem>(sizeof(typename vec_type::value_type) *
                                                         out_tensor_->mDesc.GetElementSpace());
        out_device_buffer_->SetZero();
    }

    virtual ~OpInstanceRunEngine(){};

    template <typename OpInstancePtr>
    bool Test(const std::vector<OpInstancePtr>& op_ptrs)
    {
        bool res{true};
        for(auto& op_ptr : op_ptrs)
        {
            auto invoker  = op_instance_.MakeInvokerPointer(op_ptr.get());
            auto argument = op_instance_.MakeArgumentPointer(
                op_ptr.get(), in_device_buffers_, out_device_buffer_);
            if(op_ptr->IsSupportedArgument(argument.get()))
            {
                invoker->Run(argument.get());
                out_device_buffer_->FromDevice(out_tensor_->mData.data());
                // TODO: enable flexible use of custom check_error functions
                res = res && check_err(out_tensor_->mData, ref_output_->mData);
                out_device_buffer_->SetZero();
            }
        }
        return res;
    }

    template <typename OpInstancePtr>
    ProfileBestConfig Profile(const std::vector<OpInstancePtr>& op_ptrs,
                              int nrepeat          = 100,
                              bool do_verification = false)
    {
        bool res{true};
        ProfileBestConfig best_config;

        for(auto& op_ptr : op_ptrs)
        {
            auto invoker  = op_instance_.MakeInvokerPointer(op_ptr.get());
            auto argument = op_instance_.MakeArgumentPointer(
                op_ptr.get(), in_device_buffers_, out_device_buffer_);
            if(op_ptr->IsSupportedArgument(argument.get()))
            {
                std::string op_name = op_ptr->GetTypeString();
                float avg_time      = invoker->Run(argument.get(), nrepeat);

                std::size_t flops     = op_instance_.getFlops();
                std::size_t num_btype = op_instance_.getBtype();
                float tflops          = static_cast<float>(flops) / 1.E9 / avg_time;
                float gb_per_sec      = num_btype / 1.E6 / avg_time;

                std::cout << "Perf: " << avg_time << " ms, " << tflops << " TFlops, " << gb_per_sec
                          << " GB/s, " << op_name << std::endl;

                if(tflops < best_config.best_tflops)
                {
                    best_config.best_op_name    = op_name;
                    best_config.best_tflops     = tflops;
                    best_config.best_gb_per_sec = gb_per_sec;
                    best_config.best_avg_time   = avg_time;
                }

                if(do_verification)
                {
                    out_device_buffer_->FromDevice(out_tensor_->mData.data());
                    // TODO: enable flexible use of custom check_error functions
                    res = res && CheckErr(out_tensor_->mData, ref_output_->mData);

                    // if (do_log)
                    // {

                    // }
                }
                out_device_buffer_->SetZero();
            }
        }
        return best_config;
    }

    void SetAtol(double a) { atol_ = a; }
    void SetRtol(double r) { rtol_ = r; }

    private:
    template <typename F, std::size_t... Is>
    void CallRefOpUnpackArgs(const F& f, std::index_sequence<Is...>)
    {
        f(*std::get<Is>(in_tensors_)..., *ref_output_);
    }

    const OpInstanceT& op_instance_;
    double rtol_{1e-5};
    double atol_{1e-8};

    InTensorsTuple in_tensors_;
    TensorPtr<OutDataType> out_tensor_;
    TensorPtr<OutDataType> ref_output_;

    DeviceBuffers in_device_buffers_;
    DeviceMemPtr out_device_buffer_;

    template <typename T>
    bool CheckErr(const std::vector<T>& dev_out, const std::vector<T>& ref_out) const
    {
        return ck::utils::check_err(dev_out, ref_out, "Error: incorrect results!", atol_, rtol_);
    }
};

} // namespace utils
} // namespace ck
