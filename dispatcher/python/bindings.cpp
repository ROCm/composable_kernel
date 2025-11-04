// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/// Python bindings for CK Tile Dispatcher using pybind11

#include "ck_tile/dispatcher.hpp"
#include "ck_tile/dispatcher/backends/backend_base.hpp"
#include "ck_tile/dispatcher/backends/tile_backend.hpp"
#include "ck_tile/dispatcher/backends/library_backend.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

namespace py = pybind11;
using namespace ck_tile::dispatcher;

PYBIND11_MODULE(_ck_dispatcher_cpp, m) {
    m.doc() = R"pbdoc(
        CK Tile Dispatcher C++ Extension
        ---------------------------------
        
        Low-level C++ bindings for the CK Tile GEMM dispatcher.
        
        Most users should use the high-level Python API in ck_tile_dispatcher module.
    )pbdoc";
    
    // Enums
    py::enum_<DataType>(m, "DataType")
        .value("FP16", DataType::FP16)
        .value("BF16", DataType::BF16)
        .value("FP32", DataType::FP32)
        .value("FP8", DataType::FP8)
        .value("BF8", DataType::BF8)
        .value("INT8", DataType::INT8)
        .value("INT32", DataType::INT32)
        .value("UNKNOWN", DataType::UNKNOWN)
        .export_values();
    
    py::enum_<LayoutTag>(m, "LayoutTag")
        .value("RowMajor", LayoutTag::RowMajor)
        .value("ColMajor", LayoutTag::ColMajor)
        .value("PackedExternal", LayoutTag::PackedExternal)
        .export_values();
    
    py::enum_<Pipeline>(m, "Pipeline")
        .value("Mem", Pipeline::Mem)
        .value("CompV1", Pipeline::CompV1)
        .value("CompV2", Pipeline::CompV2)
        .value("CompV3", Pipeline::CompV3)
        .value("CompV4", Pipeline::CompV4)
        .value("CompV5", Pipeline::CompV5)
        .export_values();
    
    py::enum_<Epilogue>(m, "Epilogue")
        .value("None_", Epilogue::None)
        .value("Bias", Epilogue::Bias)
        .value("Activation", Epilogue::Activation)
        .value("CShuffle", Epilogue::CShuffle)
        .value("Default", Epilogue::Default)
        .export_values();
    
    py::enum_<Scheduler>(m, "Scheduler")
        .value("Auto", Scheduler::Auto)
        .value("Intrawave", Scheduler::Intrawave)
        .value("Interwave", Scheduler::Interwave)
        .export_values();
    
    // Problem
    py::class_<Problem>(m, "Problem")
        .def(py::init<>())
        .def(py::init<std::int64_t, std::int64_t, std::int64_t>(),
             py::arg("M"), py::arg("N"), py::arg("K"))
        .def_readwrite("M", &Problem::M)
        .def_readwrite("N", &Problem::N)
        .def_readwrite("K", &Problem::K)
        .def_readwrite("k_batch", &Problem::k_batch)
        .def_readwrite("smem_budget", &Problem::smem_budget)
        .def_readwrite("prefer_persistent", &Problem::prefer_persistent)
        .def_readwrite("enable_validation", &Problem::enable_validation)
        .def("is_valid", &Problem::is_valid)
        .def("num_ops", &Problem::num_ops)
        .def("__repr__", [](const Problem& p) {
            return "<Problem M=" + std::to_string(p.M) +
                   " N=" + std::to_string(p.N) +
                   " K=" + std::to_string(p.K) + ">";
        });
    
    // KernelKey nested structs
    py::class_<KernelKey::Signature>(m, "Signature")
        .def(py::init<>())
        .def_readwrite("dtype_a", &KernelKey::Signature::dtype_a)
        .def_readwrite("dtype_b", &KernelKey::Signature::dtype_b)
        .def_readwrite("dtype_c", &KernelKey::Signature::dtype_c)
        .def_readwrite("dtype_acc", &KernelKey::Signature::dtype_acc)
        .def_readwrite("layout_a", &KernelKey::Signature::layout_a)
        .def_readwrite("layout_b", &KernelKey::Signature::layout_b)
        .def_readwrite("layout_c", &KernelKey::Signature::layout_c)
        .def_readwrite("transpose_a", &KernelKey::Signature::transpose_a)
        .def_readwrite("transpose_b", &KernelKey::Signature::transpose_b)
        .def_readwrite("grouped", &KernelKey::Signature::grouped)
        .def_readwrite("split_k", &KernelKey::Signature::split_k)
        .def_readwrite("elementwise_op", &KernelKey::Signature::elementwise_op)
        .def_readwrite("num_d_tensors", &KernelKey::Signature::num_d_tensors)
        .def_readwrite("structured_sparsity", &KernelKey::Signature::structured_sparsity);
    
    py::class_<KernelKey::Algorithm::TileShape>(m, "TileShape")
        .def(py::init<>())
        .def_readwrite("m", &KernelKey::Algorithm::TileShape::m)
        .def_readwrite("n", &KernelKey::Algorithm::TileShape::n)
        .def_readwrite("k", &KernelKey::Algorithm::TileShape::k);
    
    py::class_<KernelKey::Algorithm::WaveShape>(m, "WaveShape")
        .def(py::init<>())
        .def_readwrite("m", &KernelKey::Algorithm::WaveShape::m)
        .def_readwrite("n", &KernelKey::Algorithm::WaveShape::n)
        .def_readwrite("k", &KernelKey::Algorithm::WaveShape::k);
    
    py::class_<KernelKey::Algorithm::WarpTileShape>(m, "WarpTileShape")
        .def(py::init<>())
        .def_readwrite("m", &KernelKey::Algorithm::WarpTileShape::m)
        .def_readwrite("n", &KernelKey::Algorithm::WarpTileShape::n)
        .def_readwrite("k", &KernelKey::Algorithm::WarpTileShape::k);
    
    py::class_<KernelKey::Algorithm>(m, "Algorithm")
        .def(py::init<>())
        .def_readwrite("tile_shape", &KernelKey::Algorithm::tile_shape)
        .def_readwrite("wave_shape", &KernelKey::Algorithm::wave_shape)
        .def_readwrite("warp_tile_shape", &KernelKey::Algorithm::warp_tile_shape)
        .def_readwrite("pipeline", &KernelKey::Algorithm::pipeline)
        .def_readwrite("scheduler", &KernelKey::Algorithm::scheduler)
        .def_readwrite("epilogue", &KernelKey::Algorithm::epilogue)
        .def_readwrite("block_size", &KernelKey::Algorithm::block_size)
        .def_readwrite("double_buffer", &KernelKey::Algorithm::double_buffer)
        .def_readwrite("persistent", &KernelKey::Algorithm::persistent)
        .def_readwrite("preshuffle", &KernelKey::Algorithm::preshuffle)
        .def_readwrite("transpose_c", &KernelKey::Algorithm::transpose_c)
        .def_readwrite("num_wave_groups", &KernelKey::Algorithm::num_wave_groups);
    
    // KernelKey
    py::class_<KernelKey>(m, "KernelKey")
        .def(py::init<>())
        .def_readwrite("signature", &KernelKey::signature)
        .def_readwrite("algorithm", &KernelKey::algorithm)
        .def_readwrite("gfx_arch", &KernelKey::gfx_arch)
        .def_readwrite("structured_sparsity", &KernelKey::structured_sparsity)
        .def("encode_identifier", &KernelKey::encode_identifier)
        .def("__eq__", [](const KernelKey& a, const KernelKey& b) { return a == b; })
        .def("__ne__", [](const KernelKey& a, const KernelKey& b) { return a != b; })
        .def("__repr__", [](const KernelKey& k) {
            return "<KernelKey id='" + k.encode_identifier() + "'>";
        });
    
    // KernelInstance (abstract base)
    py::class_<KernelInstance, std::shared_ptr<KernelInstance>>(m, "KernelInstance")
        .def("get_key", &KernelInstance::get_key, py::return_value_policy::reference)
        .def("supports", &KernelInstance::supports)
        .def("get_name", &KernelInstance::get_name)
        // Note: run() and validate() require device pointers, typically not called from Python
        .def("__repr__", [](const KernelInstance& k) {
            return "<KernelInstance name='" + k.get_name() + "'>";
        });
    
    // Registry
    py::enum_<Registry::Priority>(m, "Priority")
        .value("Low", Registry::Priority::Low)
        .value("Normal", Registry::Priority::Normal)
        .value("High", Registry::Priority::High)
        .export_values();
    
    py::class_<Registry>(m, "Registry")
        .def_static("instance", &Registry::instance, py::return_value_policy::reference)
        .def("register_kernel", &Registry::register_kernel,
             py::arg("instance"), py::arg("priority") = Registry::Priority::Normal)
        .def("lookup", py::overload_cast<const std::string&>(&Registry::lookup, py::const_))
        .def("lookup", py::overload_cast<const KernelKey&>(&Registry::lookup, py::const_))
        .def("get_all", &Registry::get_all)
        .def("filter", &Registry::filter)
        .def("size", &Registry::size)
        .def("clear", &Registry::clear)
        .def("__len__", &Registry::size)
        .def("__repr__", [](const Registry& r) {
            return "<Registry size=" + std::to_string(r.size()) + ">";
        });
    
    // Dispatcher
    py::enum_<Dispatcher::SelectionStrategy>(m, "SelectionStrategy")
        .value("FirstFit", Dispatcher::SelectionStrategy::FirstFit)
        .value("Heuristic", Dispatcher::SelectionStrategy::Heuristic)
        .export_values();
    
    py::class_<Dispatcher>(m, "Dispatcher")
        .def(py::init<>())
        .def(py::init<Registry*>())
        .def("set_heuristic", &Dispatcher::set_heuristic)
        .def("set_strategy", &Dispatcher::set_strategy)
        .def("select_kernel", &Dispatcher::select_kernel)
        // Note: run() methods require device pointers, typically called from C++ side
        .def("__repr__", []() {
            return "<Dispatcher>";
        });
    
    // Backend types
    py::enum_<backends::BackendType>(m, "BackendType")
        .value("Tile", backends::BackendType::Tile)
        .value("Library", backends::BackendType::Library)
        .value("JIT", backends::BackendType::JIT)
        .value("Unknown", backends::BackendType::Unknown)
        .export_values();
    
    // KernelInstance (abstract base)
    py::class_<backends::KernelInstance, std::shared_ptr<backends::KernelInstance>>(m, "KernelInstanceCpp")
        .def("get_key", &backends::KernelInstance::get_key, py::return_value_policy::reference)
        .def("supports", &backends::KernelInstance::supports)
        .def("get_name", &backends::KernelInstance::get_name)
        .def("get_backend_type", &backends::KernelInstance::get_backend_type)
        .def("get_metadata", &backends::KernelInstance::get_metadata)
        .def("run", [](backends::KernelInstance& self,
                      std::uintptr_t a_ptr,
                      std::uintptr_t b_ptr,
                      std::uintptr_t c_ptr,
                      const Problem& problem,
                      std::uintptr_t stream_ptr) {
            return self.run(reinterpret_cast<const void*>(a_ptr),
                          reinterpret_cast<const void*>(b_ptr),
                          reinterpret_cast<void*>(c_ptr),
                          problem,
                          reinterpret_cast<hipStream_t>(stream_ptr));
        }, py::arg("a_ptr"), py::arg("b_ptr"), py::arg("c_ptr"), 
           py::arg("problem"), py::arg("stream_ptr") = 0)
        .def("__repr__", [](const backends::KernelInstance& k) {
            return "<KernelInstanceCpp name='" + k.get_name() + "'>";
        });
    
    // TileBackend
    py::class_<backends::TileBackend>(m, "TileBackendCpp")
        .def(py::init<>())
        .def("discover_kernels", &backends::TileBackend::discover_kernels)
        .def("get_backend_type", &backends::TileBackend::get_backend_type)
        .def("__repr__", []() {
            return "<TileBackendCpp>";
        });
    
    // LibraryBackend
    py::class_<backends::LibraryBackend>(m, "LibraryBackendCpp")
        .def(py::init<>())
        .def("discover_kernels", &backends::LibraryBackend::discover_kernels)
        .def("enumerate_operations", &backends::LibraryBackend::enumerate_operations)
        .def("get_backend_type", &backends::LibraryBackend::get_backend_type)
        .def("__repr__", []() {
            return "<LibraryBackendCpp>";
        });
}


