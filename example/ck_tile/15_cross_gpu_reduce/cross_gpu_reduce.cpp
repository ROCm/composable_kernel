// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include <hip/hip_runtime.h>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <future>
#include <vector>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsuggest-destructor-override"
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wshadow-field-in-constructor"
#pragma clang diagnostic ignored "-Wdocumentation"
#pragma clang diagnostic ignored "-Winconsistent-missing-destructor-override"
#pragma clang diagnostic ignored "-Wcast-align"
#pragma clang diagnostic ignored "-Wglobal-constructors"
#pragma clang diagnostic ignored "-Wdeprecated-copy-with-user-provided-dtor"

#include <mscclpp/core.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <mscclpp/sm_channel.hpp>
#include <mscclpp/semaphore.hpp>

#pragma clang diagnostic pop

#include "cross_gpu_reduce.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/cross_gpu_reduce.hpp"

template <class T>
using DeviceHandle = mscclpp::DeviceHandle<T>;
extern __constant__ DeviceHandle<mscclpp::SmChannel> constSlaveSmChannels[8]; // For SmChannel

extern __constant__ DeviceHandle<mscclpp::SmChannel> constMasterSmChannel;

void setupConnection(int rank, int slaveRank, int worldSize, void* dst_data, size_t dataSize)
{
    // Initialize MSCCL++ Communicator
    auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(rank, worldSize);

    mscclpp::Communicator comm(bootstrap);
    mscclpp::Transport transport = mscclpp::Transport::CudaIpc;

    // We'll register our local memory. For the slave, this might be the destination buffer.
    // For senders, this might be the source buffer or a local buffer we expose to the slave.
    mscclpp::RegisteredMemory localMemory = comm.registerMemory(dst_data, dataSize, transport);

    if(rank == slaveRank)
    {
        std::vector<mscclpp::NonblockingFuture<std::shared_ptr<mscclpp::Connection>>>
            connectionFutures;
        std::vector<mscclpp::NonblockingFuture<mscclpp::RegisteredMemory>> remoteMemFutures;
        std::vector<std::shared_ptr<mscclpp::SmDevice2DeviceSemaphore>> slave_semaphore_list(
            worldSize);
        for(size_t senderRank = 0; senderRank < static_cast<size_t>(worldSize); ++senderRank)
        {
            if(senderRank == static_cast<size_t>(rank))
                continue;
            connectionFutures.push_back(comm.connectOnSetup(senderRank, 0, transport));
            comm.sendMemoryOnSetup(localMemory, senderRank, 0);
            remoteMemFutures.push_back(comm.recvMemoryOnSetup(senderRank, 0));
        }
        comm.setup();
        // Now retrieve all completed futures
        std::vector<std::shared_ptr<mscclpp::Connection>> connections;
        connections.reserve(connectionFutures.size());
        for(auto& cf : connectionFutures)
        {
            connections.push_back(cf.get());
        }

        std::vector<mscclpp::RegisteredMemory> remoteMemories;
        remoteMemories.reserve(remoteMemFutures.size());
        for(auto& rmf : remoteMemFutures)
        {
            remoteMemories.push_back(rmf.get());
        }

        // Create semaphores and channels
        // One semaphore per connection
        std::vector<std::shared_ptr<mscclpp::SmDevice2DeviceSemaphore>> slaveSemaphores;
        slaveSemaphores.reserve(connections.size());
        for(auto& conn : connections)
        {
            slaveSemaphores.push_back(
                std::make_shared<mscclpp::SmDevice2DeviceSemaphore>(comm, conn));
        }

        // Create channels
        std::vector<DeviceHandle<mscclpp::SmChannel>> SmChannels;
        SmChannels.reserve(slaveSemaphores.size());
        for(size_t i = 0; i < slaveSemaphores.size(); ++i)
        {
            SmChannels.push_back(mscclpp::deviceHandle(
                mscclpp::SmChannel(slaveSemaphores[i],
                                   remoteMemories[i], // Remote buffer from the sender
                                   dst_data           // Local buffer (this slave's buffer)
                                   )));
        }
        hipError_t error_slave =
            hipMemcpyToSymbol(constSlaveSmChannels,
                              SmChannels.data(),
                              sizeof(DeviceHandle<mscclpp::SmChannel>) * SmChannels.size());
        if(error_slave != hipSuccess)
        {
            std::cerr << "Error locating data to constant memory" << std::endl;
            return;
        }
    }
    else
    {
        // This is a sender:
        // We only connect to the slave, send our memory handle, and receive the slave's memory
        // handle.
        mscclpp::NonblockingFuture<std::shared_ptr<mscclpp::Connection>> connectionFuture =
            comm.connectOnSetup(slaveRank, 0, transport);
        // Send our memory to the slave
        comm.sendMemoryOnSetup(localMemory, slaveRank, 0);

        // Receive slave's memory
        mscclpp::NonblockingFuture<mscclpp::RegisteredMemory> remoteMemoryFuture =
            comm.recvMemoryOnSetup(slaveRank, 0);
        comm.setup();
        std::shared_ptr<mscclpp::Connection> connection = connectionFuture.get();
        mscclpp::RegisteredMemory remoteMemory          = remoteMemoryFuture.get();

        auto senderSemaphore =
            std::make_shared<mscclpp::SmDevice2DeviceSemaphore>(comm, connection);

        auto senderChannel = mscclpp::SmChannel(senderSemaphore, localMemory, remoteMemory.data());
        DeviceHandle<mscclpp::SmChannel> senderSmChannel = mscclpp::deviceHandle(senderChannel);

        hipError_t error_master = hipMemcpyToSymbol(
            constMasterSmChannel, &senderSmChannel, sizeof(DeviceHandle<mscclpp::SmChannel>));
        if(error_master != hipSuccess)
        {
            std::cerr << "Error locating data to constant memory" << std::endl;
            return;
        }
    }
}

template <typename InputType, typename OutputType>
struct AllocateAndTransferFunctor
{
    // Invoke the memory transfer between GPUs based on whether it is host gpu or slave gpu.
    float invoke_transfer(ck_tile::DeviceMem& transfer_buf,
                          ck_tile::index_t host_gpu,
                          int device_id,
                          const ck_tile::ArgParser& arg_parser,
                          const ck_tile::stream_config& s,
                          std::promise<const void*>& host_receive_ptr_promise,
                          std::future<const void*>& host_receive_ptr_future)
    {
        ck_tile::index_t M = arg_parser.get_int("M");
        ck_tile::index_t N = arg_parser.get_int("N");

        constexpr ck_tile::index_t M_Tile = 128;
        constexpr ck_tile::index_t N_Tile = 128;

        constexpr ck_tile::index_t M_Warp = 2;
        constexpr ck_tile::index_t N_Warp = 2;

        constexpr ck_tile::index_t M_Warp_Tile = 64;
        constexpr ck_tile::index_t N_Warp_Tile = 64;

        constexpr int kBlockPerCu = 1;
        using Vector              = ck_tile::sequence<8, 8>;

        using ReduceShape = ck_tile::TileReduceShape<ck_tile::sequence<M_Tile, N_Tile>,
                                                     ck_tile::sequence<M_Warp, N_Warp>,
                                                     ck_tile::sequence<M_Warp_Tile, N_Warp_Tile>,
                                                     Vector>;

        using ReducePartitioner = ck_tile::CrossReducePartitioner<ReduceShape>;

        using CrossReduceReceivePipelinePolicy = ck_tile::ReduceReceivePipelineDefaultPolicy;

        using CrossReduceSendPipelinePolicy = ck_tile::ReduceSendPipelineDefaultPolicy;

        using CrossReduceReceivePipeline =
            ck_tile::CrossReduceReceivePipelineScaleUp<InputType,
                                                       OutputType,
                                                       ReduceShape,
                                                       CrossReduceReceivePipelinePolicy>;
        using CrossReduceSendPipeline = ck_tile::
            CrossReduceSendPipelineScaleUp<InputType, ReduceShape, CrossReduceSendPipelinePolicy>;

        constexpr ck_tile::index_t kBlockSize = CrossReduceReceivePipeline::BlockSize;

        transfer_receive_basic_args args_receive;

        args_receive.p_reduce  = transfer_buf.GetDeviceBuffer();
        args_receive.host_gpu  = host_gpu;
        args_receive.device_id = static_cast<ck_tile::index_t>(device_id);
        args_receive.M         = M;
        args_receive.N         = N;

        transfer_send_basic_args args_send;
        args_send.p_reduce  = transfer_buf.GetDeviceBuffer();
        args_send.host_gpu  = host_gpu;
        args_send.device_id = static_cast<ck_tile::index_t>(device_id);
        args_send.M         = M;
        args_send.N         = N;

        float ave_time = 0.0;

        // using MasterKernel = ck_tile::ReduceSendKernel<CrossReduceSendPipeline>;
        using SlaveKernel =
            ck_tile::ReduceReceiveKernel<ReducePartitioner, CrossReduceReceivePipeline>;
        using MasterKernel = ck_tile::ReduceSendKernel<ReducePartitioner, CrossReduceSendPipeline>;
        // Depending on whether to enable the receiving kernel or sending kernel
        if(static_cast<ck_tile::index_t>(device_id) == host_gpu)
        {
            // initialize the receive data buffer and global memory location.
            ck_tile::HostTensor<InputType> receive_host({M, N});
            ck_tile::DeviceMem receive_buf(receive_host.get_element_space_size_in_bytes());
            args_receive.p_receive = receive_buf.GetDeviceBuffer();
            // initialize the output data buffer.
            std::string output_type = arg_parser.get_str("output_type");
            if(output_type.compare("float") == 0)
            {
                ck_tile::HostTensor<OutputType> output_host({M, N});
                ck_tile::DeviceMem output_buf(output_host.get_element_space_size_in_bytes());
                args_receive.p_output = output_buf.GetDeviceBuffer();
                host_receive_ptr_promise.set_value(args_receive.p_receive);
                auto kargs_slave       = SlaveKernel::MakeKargs(args_receive.p_reduce,
                                                          args_receive.p_receive,
                                                          args_receive.p_output,
                                                          args_receive.M,
                                                          args_receive.N);
                const dim3 grids_slave = SlaveKernel::GridSize(M, N);
                ave_time               = ck_tile::launch_kernel(
                    s,
                    ck_tile::make_kernel<kBlockSize, kBlockPerCu>(
                        SlaveKernel{}, grids_slave, kBlockSize, 0, kargs_slave));
            }
            else
            {
                std::cerr << "Currently, we do not support other output data type." << std::endl;
                return -1;
            }
        }
        else
        {
            const void* send_location_ptr = host_receive_ptr_future.get();
            args_send.p_send              = send_location_ptr;
            auto kargs_master             = MasterKernel::MakeKargs(
                args_send.p_reduce, args_send.p_send, args_send.M, args_send.N);
            const dim3 grids_master = MasterKernel::GridSize(M, N);
            ave_time                = ck_tile::launch_kernel(
                s,
                ck_tile::make_kernel<kBlockSize, kBlockPerCu>(
                    MasterKernel{}, grids_master, kBlockSize, 0, kargs_master));
        }

        std::string op_name{"Cross GPU Reduce"};
        std::cout << "Run" << op_name << "kernel with M =" << M << "N =" << N << " : " << ave_time
                  << "ms" << std::endl;

        return ave_time;
    }

    void operator()(int device_id,
                    ck_tile::HostTensor<InputType>& host_tensor,
                    ck_tile::DeviceMem& device_mem,
                    ck_tile::index_t host_gpu,
                    const ck_tile::ArgParser& arg_parser,
                    std::promise<const void*>& host_receive_ptr_promise,
                    std::future<const void*>& host_receive_ptr_future)
    {
        hipError_t hip_err_set_device = hipSetDevice(device_id);
        if(hip_err_set_device != hipSuccess)
        {
            std::cerr << "Error setting device " << device_id << ": "
                      << hipGetErrorString(hip_err_set_device) << std::endl;
            return;
        }
        // Allocate device memory
        device_mem.Realloc(host_tensor.get_element_space_size_in_bytes());
        // Transfer data to device
        device_mem.ToDevice(host_tensor.data());

        int worldSize = arg_parser.get_int("gpu_nums");
        setupConnection(device_id,
                        static_cast<int>(host_gpu),
                        static_cast<int>(worldSize),
                        device_mem.GetDeviceBuffer(),
                        host_tensor.get_element_space_size_in_bytes());

        int n_warmup = arg_parser.get_int("warmup");
        int n_repeat = arg_parser.get_int("repeat");

        invoke_transfer(device_mem,
                        host_gpu,
                        device_id,
                        arg_parser,
                        ck_tile::stream_config{nullptr, true, 1, n_warmup, n_repeat},
                        host_receive_ptr_promise,
                        host_receive_ptr_future);
    }
};

template <typename InputType, typename OutputType>
bool run_cross_gpu_reduce(ck_tile::ArgParser arg_parser)
{
    ck_tile::index_t gpu_nums      = arg_parser.get_int("gpu_nums");
    ck_tile::index_t host_gpu      = arg_parser.get_int("host_gpu");
    ck_tile::index_t transfer_dim1 = arg_parser.get_int("M");
    ck_tile::index_t transfer_dim2 = arg_parser.get_int("N");

    // Validate arguments
    if(gpu_nums < 1)
    {
        std::cerr << "Invalid number of GPUs specified." << std::endl;
        return -1;
    }
    // Examine how many gpus inside the server system.
    int device_count                = 0;
    hipError_t hip_err_device_count = hipGetDeviceCount(&device_count);
    if(hip_err_device_count != hipSuccess)
    {
        std::cerr << "Error getting device count: " << hipGetErrorString(hip_err_device_count)
                  << std::endl;
        return -1;
    }

    // Make sure the gpus is larger or equals to the required gpu_nums.
    if(device_count < gpu_nums)
    {
        std::cerr << "The available GPUs in the system is less than required. All available GPUs: "
                  << device_count << std::endl;
    }

    if(host_gpu < 0 || host_gpu >= device_count)
    {
        std::cerr << "Invalid host GPU index specified. Using GPU 0 as host GPU." << std::endl;
        host_gpu = 0;
    }

    // Make sure that we could open each one of the GPU.
    // Print device properties
    for(int i = 0; i < gpu_nums; ++i)
    {
        hipDeviceProp_t device_prop;
        hipError_t hip_err_device_prop = hipGetDeviceProperties(&device_prop, i);
        if(hip_err_device_prop != hipSuccess)
        {
            std::cerr << "Error getting device properties for device " << i << ": "
                      << hipGetErrorString(hip_err_device_prop) << std::endl;
            return -1;
        }
        std::cout << "GPU " << i << ": " << device_prop.name << std::endl;
    }

    std::vector<int> device_list(gpu_nums);
    std::vector<ck_tile::HostTensor<InputType>> transfer_tensor_host_list;
    transfer_tensor_host_list.reserve(gpu_nums);
    std::vector<ck_tile::DeviceMem> transfer_bufs(gpu_nums);
    std::vector<std::thread> threads;

    AllocateAndTransferFunctor<InputType, OutputType> allocateAndTransfer;

    // Initialize host tensors
    for(int i = 0; i < gpu_nums; ++i)
    {
        device_list[i]               = i; // Adjust based on available GPUs
        std::vector<int> tensor_dims = {transfer_dim1, transfer_dim2};
        transfer_tensor_host_list.emplace_back(tensor_dims);
        ck_tile::FillUniformDistribution<InputType>{-5.f, 5.f}(transfer_tensor_host_list.back());
        // Enable P2P access between GPUs
        if(i != host_gpu)
        {
            int canAccessPeer = 0;
            hipError_t err_peer =
                hipDeviceCanAccessPeer(&canAccessPeer, device_list[host_gpu], device_list[i]);
            if(err_peer != hipSuccess || !canAccessPeer)
            {
                std::cerr << "P2P not supported between device " << device_list[host_gpu]
                          << " and device " << device_list[i] << std::endl;
                return -1; // Handle error accordingly.
            }
            else
            {
                // Enable P2P access from host GPU to device i.
                hipError_t hip_err_set_device_host = hipSetDevice(device_list[host_gpu]);
                if(hip_err_set_device_host != hipSuccess)
                {
                    std::cerr << "Error setting the host device " << host_gpu << ": "
                              << hipGetErrorString(hip_err_set_device_host) << std::endl;
                    return -1;
                }
                hipError_t err_peer_host = hipDeviceEnablePeerAccess(device_list[i], 0);
                if(err_peer_host != hipSuccess && err_peer_host != hipErrorPeerAccessAlreadyEnabled)
                {
                    std::cerr << "Error enabling peer access from host " << device_list[host_gpu]
                              << " to device " << device_list[i] << ": "
                              << hipGetErrorString(err_peer_host) << std::endl;
                    return -1;
                }
                // Enable P2P access from device i to host GPU.
                hipError_t hip_err_set_device_send = hipSetDevice(device_list[i]);
                if(hip_err_set_device_send != hipSuccess)
                {
                    std::cerr << "Error setting the host device " << host_gpu << ": "
                              << hipGetErrorString(hip_err_set_device_send) << std::endl;
                    return -1;
                }
                hipError_t err_peer_device = hipDeviceEnablePeerAccess(device_list[host_gpu], 0);
                if(err_peer_device != hipSuccess &&
                   err_peer_device != hipErrorPeerAccessAlreadyEnabled)
                {
                    std::cerr << "Error enabling peer access from device " << device_list[i]
                              << " to device " << device_list[host_gpu] << ": "
                              << hipGetErrorString(err_peer_device) << std::endl;
                    return -1;
                }
            }
        }
    }

    for(int i = 0; i < gpu_nums; ++i)
    {
        hipError_t hip_device_sync_enable = hipSetDevice(device_list[i]);
        if(hip_device_sync_enable != hipSuccess)
        {
            std::cerr << "Error enable the device for synchronization" << std::endl;
            return -1;
        }
        hipError_t hip_device_sync_err = hipDeviceSynchronize();
        if(hip_device_sync_err != hipSuccess)
        {
            std::cerr << "Error in complete the device for synchronization" << std::endl;
            return -1;
        }
    }

    std::promise<const void*> host_receive_ptr_promise;
    std::future<const void*> host_receive_ptr_future = host_receive_ptr_promise.get_future();

    for(int i = 0; i < gpu_nums; ++i)
    {
        threads.emplace_back(allocateAndTransfer,
                             device_list[i],
                             std::ref(transfer_tensor_host_list[i]),
                             std::ref(transfer_bufs[i]),
                             host_gpu,
                             arg_parser,
                             std::ref(host_receive_ptr_promise),
                             std::ref(host_receive_ptr_future));
    }

    // Wait for all threads to complete
    for(auto& t : threads)
    {
        t.join();
    }

    bool pass = true;
    return !pass;
}

int main(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;
    std::string prec = arg_parser.get_str("pr");
    bool run_result  = true;
    if(prec.compare("fp16") == 0)
    {
        run_result &= run_cross_gpu_reduce<ck_tile::fp16_t, float>(arg_parser);
    }

    return run_result ? 0 : 1;
}
