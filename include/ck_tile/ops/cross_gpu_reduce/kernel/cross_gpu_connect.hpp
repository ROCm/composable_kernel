// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once
#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"

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
