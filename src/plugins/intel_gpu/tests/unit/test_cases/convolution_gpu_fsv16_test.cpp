// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "random_generator.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/convolution.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/reorder.hpp>

#include <cmath>
#include <string>
#include <vector>

using namespace cldnn;
using namespace ::tests;

namespace {
const std::string no_bias = "";

struct oob_guard_experiment_test_params {
    std::string name;
    ov::Shape in_shape;
    ov::Shape wei_shape;
    ov::Strides stride;
    ov::CoordinateDiff pad_begin;
    ov::CoordinateDiff pad_end;
    int input_x_pad_after;
    bool is_1x1;
    bool safe_no_oob_by_params;
};

size_t calc_output_dim(size_t in,
                       size_t kernel,
                       size_t stride,
                       size_t dilation,
                       ptrdiff_t pad_begin,
                       ptrdiff_t pad_end) {
    const size_t kernel_extent = dilation * (kernel - 1) + 1;
    return (in + static_cast<size_t>(pad_begin + pad_end) - kernel_extent) / stride + 1;
}
}  // namespace

class convolution_gpu_bfyx_f16_oob : public testing::TestWithParam<oob_guard_experiment_test_params> {};

// Test matrix for discussion from mails:
// - classify host-side cases where OOB is not expected upfront (safe_no_oob_by_params=true)
// - include potentially unsafe/tail/input-padding cases
// - compare forced optimized kernel against convolution_gpu_ref
TEST_P(convolution_gpu_bfyx_f16_oob, forced_kernel_compare_with_ref) {
    auto& engine = get_test_engine();
    const auto p = GetParam();

    if (!engine.get_device_info().supports_fp16)
        GTEST_SKIP() << "fp16 not supported on this device.";

    auto input_mem = engine.allocate_memory({p.in_shape, data_types::f16, format::bfyx});
    auto weights = engine.allocate_memory({p.wei_shape, data_types::f16, format::bfyx});

    tests::random_generator rg(GET_SUITE_NAME);
    set_values(input_mem, rg.generate_random_1d<ov::float16>(ov::shape_size(p.in_shape), -5, 5));
    set_values(weights, rg.generate_random_1d<ov::float16>(ov::shape_size(p.wei_shape), -5, 5));

    const size_t out_x = calc_output_dim(p.in_shape[3],
                                         p.wei_shape[3],
                                         p.stride[1],
                                         1,
                                         p.pad_begin[1],
                                         p.pad_end[1]);
    const bool likely_tail_case = (out_x % 8) != 0;

    SCOPED_TRACE("case=" + p.name +
                 ", out_x=" + std::to_string(out_x) +
                 ", likely_tail_case=" + std::to_string(likely_tail_case) +
                 ", input_x_pad_after=" + std::to_string(p.input_x_pad_after) +
                 ", safe_no_oob_by_params=" + std::to_string(p.safe_no_oob_by_params));

    layout dyn_input_layout{ov::PartialShape::dynamic(4), data_types::f16, format::bfyx};
    layout input_fsv16_layout{p.in_shape, data_types::f16, format::b_fs_yx_fsv16};
    if (p.input_x_pad_after > 0) {
        padding x_padding({0, 0, 0, 0}, {0, 0, 0, p.input_x_pad_after});
        input_fsv16_layout = layout(data_types::f16,
                                    format::b_fs_yx_fsv16,
                                    tensor{static_cast<int32_t>(p.in_shape[0]),
                                           static_cast<int32_t>(p.in_shape[1]),
                                           static_cast<int32_t>(p.in_shape[3]),
                                           static_cast<int32_t>(p.in_shape[2])},
                                    x_padding);
    }

    auto make_topology = [&]() {
        return topology(
            input_layout("input", dyn_input_layout),
            reorder("input_fsv16", input_info("input"), input_fsv16_layout),
            data("weights", weights),
            convolution("conv", input_info("input_fsv16"), "weights", no_bias, 1,
                        p.stride, ov::Strides{1, 1}, p.pad_begin, p.pad_end, false));
    };

    const std::string kernel_name = p.is_1x1 ? "convolution_gpu_bfyx_f16_1x1" : "convolution_gpu_bfyx_f16";

    ExecutionConfig cfg_test = get_test_default_config(engine);
    cfg_test.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{
        {"conv", {format::b_fs_yx_fsv16, kernel_name, impl_types::ocl}}
    }));
    cfg_test.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    ExecutionConfig cfg_ref = get_test_default_config(engine);
    cfg_ref.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{
        {"conv", {format::b_fs_yx_fsv16, "convolution_gpu_ref", impl_types::ocl}}
    }));
    cfg_ref.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    network net_test(engine, make_topology(), cfg_test);
    network net_ref(engine, make_topology(), cfg_ref);

    net_test.set_input_data("input", input_mem);
    net_ref.set_input_data("input", input_mem);

    auto out_test = net_test.execute();
    auto out_ref = net_ref.execute();

    auto mem_test = out_test.at("conv").get_memory();
    auto mem_ref = out_ref.at("conv").get_memory();

    cldnn::mem_lock<ov::float16> ptr_test(mem_test, get_test_stream());
    cldnn::mem_lock<ov::float16> ptr_ref(mem_ref, get_test_stream());

    ASSERT_EQ(mem_test->get_layout(), mem_ref->get_layout());
    ASSERT_EQ(ptr_test.size(), ptr_ref.size());

    const float atol = 0.1f;
    const float rtol = 1e-2f;
    for (size_t i = 0; i < ptr_test.size(); i++) {
        const float test_val = float(ptr_test[i]);
        const float ref_val = float(ptr_ref[i]);
        const float tol = atol + rtol * std::fabs(ref_val);
        ASSERT_NEAR(test_val, ref_val, tol) << "Mismatch at idx=" << i;
    }
}

INSTANTIATE_TEST_SUITE_P(
    convolution_gpu_bfyx_f16_experiments,
    convolution_gpu_bfyx_f16_oob,
    testing::ValuesIn(std::vector<oob_guard_experiment_test_params>{
        // Upfront safe candidates (no input padding + no tail in X)
        // out_x divisible by 8 (blockWidth=8) – definitely no tail
        {"safe_1x1_out_x_div8", {1, 32, 8, 16}, {32, 32, 1, 1}, ov::Strides{1, 1}, ov::CoordinateDiff{0, 0}, ov::CoordinateDiff{0, 0}, 0, true, true},
        {"safe_3x3_out_x_div8", {1, 32, 8, 16}, {32, 32, 3, 3}, ov::Strides{1, 1}, ov::CoordinateDiff{0, 0}, ov::CoordinateDiff{0, 0}, 0, false, true},
        // out_x divisible by 4 but not 8 (blockWidth=4)
        {"safe_1x1_out_x_div4", {1, 32, 8, 12}, {32, 32, 1, 1}, ov::Strides{1, 1}, ov::CoordinateDiff{0, 0}, ov::CoordinateDiff{0, 0}, 0, true, true},
        {"safe_3x3_out_x_div4", {1, 32, 8, 12}, {32, 32, 3, 3}, ov::Strides{1, 1}, ov::CoordinateDiff{0, 0}, ov::CoordinateDiff{0, 0}, 0, false, true},
        // larger features (64) – out_x divisible by 8
        {"safe_1x1_no_padding_divisible_x_batch2", {2, 32, 6, 24}, {32, 32, 1, 1}, ov::Strides{1, 1}, ov::CoordinateDiff{0, 0}, ov::CoordinateDiff{0, 0}, 0, true, true},
        {"safe_3x3_no_padding_divisible_x_batch2", {2, 32, 6, 24}, {32, 32, 3, 3}, ov::Strides{1, 1}, ov::CoordinateDiff{0, 0}, ov::CoordinateDiff{0, 0}, 0, false, true},
        {"safe_5x5_out_x_div4", {1, 32, 12, 20}, {32, 32, 5, 5}, ov::Strides{1, 1}, ov::CoordinateDiff{0, 0}, ov::CoordinateDiff{0, 0}, 0, false, true},

        // Potentially unsafe candidates (tail and/or padded input)
        // -- tail in X not divisible by blockWidth=2 (out_x=15)
        {"potential_1x1_tail_x_mod2", {1, 32, 7, 15}, {32, 32, 1, 1}, ov::Strides{1, 1}, ov::CoordinateDiff{0, 0}, ov::CoordinateDiff{0, 0}, 0, true, false},
        {"potential_3x3_tail_x_mod2", {1, 32, 7, 15}, {32, 32, 3, 3}, ov::Strides{1, 1}, ov::CoordinateDiff{0, 0}, ov::CoordinateDiff{0, 0}, 0, false, false},
        // -- tail in X not divisible by blockWidth=4 (out_x=13)
        {"potential_1x1_tail_x_mod4", {1, 32, 9, 13}, {32, 32, 1, 1}, ov::Strides{1, 1}, ov::CoordinateDiff{0, 0}, ov::CoordinateDiff{0, 0}, 0, true, false},
        {"potential_3x3_tail_x_mod4", {1, 32, 9, 13}, {32, 32, 3, 3}, ov::Strides{1, 1}, ov::CoordinateDiff{0, 0}, ov::CoordinateDiff{0, 0}, 0, false, false},
        // -- tail in X = 1 (out_x=1 – smallest possible output tile)
        {"potential_1x1_out_x_1", {1, 32, 4, 1}, {32, 32, 1, 1}, ov::Strides{1, 1}, ov::CoordinateDiff{0, 0}, ov::CoordinateDiff{0, 0}, 0, true, false},
        {"potential_3x3_out_x_1", {1, 32, 4, 3}, {32, 32, 3, 3}, ov::Strides{1, 1}, ov::CoordinateDiff{0, 0}, ov::CoordinateDiff{0, 0}, 0, false, false},
        // -- input padding only on right (asymmetric: pad_before=0, pad_after>0)
        {"potential_1x1_input_padding", {1, 32, 7, 16}, {32, 32, 1, 1}, ov::Strides{1, 1}, ov::CoordinateDiff{0, 0}, ov::CoordinateDiff{0, 0}, 4, true, false},
        {"potential_3x3_input_padding", {1, 32, 7, 16}, {32, 32, 3, 3}, ov::Strides{1, 1}, ov::CoordinateDiff{0, 0}, ov::CoordinateDiff{0, 0}, 4, false, false},
        // -- input padding + tail (both risks combined)
        {"potential_1x1_input_padding_tail", {1, 32, 7, 15}, {32, 32, 1, 1}, ov::Strides{1, 1}, ov::CoordinateDiff{0, 0}, ov::CoordinateDiff{0, 0}, 3, true, false},
        {"potential_3x3_input_padding_tail", {1, 32, 7, 15}, {32, 32, 3, 3}, ov::Strides{1, 1}, ov::CoordinateDiff{0, 0}, ov::CoordinateDiff{0, 0}, 3, false, false},
        // -- batch=2 + input padding
        {"potential_1x1_input_padding_batch2", {2, 32, 9, 16}, {32, 32, 1, 1}, ov::Strides{1, 1}, ov::CoordinateDiff{0, 0}, ov::CoordinateDiff{0, 0}, 3, true, false},
        {"potential_3x3_input_padding_batch2", {2, 32, 9, 16}, {32, 32, 3, 3}, ov::Strides{1, 1}, ov::CoordinateDiff{0, 0}, ov::CoordinateDiff{0, 0}, 3, false, false},
        // -- 5x5 kernel: tail
        {"potential_5x5_tail_x", {1, 32, 10, 19}, {32, 32, 5, 5}, ov::Strides{1, 1}, ov::CoordinateDiff{0, 0}, ov::CoordinateDiff{0, 0}, 0, false, false},
        {"potential_5x5_input_padding", {1, 32, 10, 20}, {32, 32, 5, 5}, ov::Strides{1, 1}, ov::CoordinateDiff{0, 0}, ov::CoordinateDiff{0, 0}, 5, false, false},
    }),
    [](const testing::TestParamInfo<oob_guard_experiment_test_params>& info) {
        return info.param.name;
    }
);

// Stress test configuration: various input/kernel/batch shapes.
struct stress_test_config {
    std::string name;
    ov::Shape in_shape;
    ov::Shape wei_shape;
    int iterations;
};

class convolution_gpu_bfyx_f16_stress : public testing::TestWithParam<stress_test_config> {};

// Reproduction-oriented stress test for runtime DEVICE_LOST-like failures.
// Explores various tensor dimensions, kernel sizes, and batch configurations
// to increase probability of surfacing GPU runtime instability.
TEST_P(convolution_gpu_bfyx_f16_stress, repeated_execute_stability) {
    auto& engine = get_test_engine();

    if (!engine.get_device_info().supports_fp16)
        GTEST_SKIP() << "fp16 not supported on this device.";

    const auto cfg = GetParam();
    auto input_mem = engine.allocate_memory({cfg.in_shape, data_types::f16, format::bfyx});
    auto weights = engine.allocate_memory({cfg.wei_shape, data_types::f16, format::bfyx});

    tests::random_generator rg(GET_SUITE_NAME);
    set_values(input_mem, rg.generate_random_1d<ov::float16>(ov::shape_size(cfg.in_shape), -3, 3));
    set_values(weights, rg.generate_random_1d<ov::float16>(ov::shape_size(cfg.wei_shape), -3, 3));

    layout dyn_input_layout{ov::PartialShape::dynamic(4), data_types::f16, format::bfyx};
    layout input_fsv16_layout{cfg.in_shape, data_types::f16, format::b_fs_yx_fsv16};

    topology topo(
        input_layout("input", dyn_input_layout),
        reorder("input_fsv16", input_info("input"), input_fsv16_layout),
        data("weights", weights),
        convolution("conv", input_info("input_fsv16"), "weights", no_bias, 1,
                    ov::Strides{1, 1}, ov::Strides{1, 1}, ov::CoordinateDiff{0, 0}, ov::CoordinateDiff{0, 0}, false));

    ExecutionConfig cfg_exec = get_test_default_config(engine);
    cfg_exec.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{
        {"conv", {format::b_fs_yx_fsv16, "convolution_gpu_bfyx_f16", impl_types::ocl}}
    }));
    cfg_exec.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    network net(engine, topo, cfg_exec);

    SCOPED_TRACE("case=" + cfg.name + ", iterations=" + std::to_string(cfg.iterations));

    std::string error_msg;
    
    try {
        for (int iter = 0; iter < cfg.iterations; iter++) {
            net.set_input_data("input", input_mem);
            auto out = net.execute();
            auto mem = out.at("conv").get_memory();
            cldnn::mem_lock<ov::float16> ptr(mem, get_test_stream());
            EXPECT_GT(ptr.size(), size_t(0)) << "Empty output at iteration " << iter;
        }
    } catch (const std::exception& e) {
        error_msg = e.what();
        ADD_FAILURE() << "Exception in test case '" << cfg.name << "': " << error_msg;
    }
}

INSTANTIATE_TEST_SUITE_P(
    stress_dimensions,
    convolution_gpu_bfyx_f16_stress,
    testing::ValuesIn(std::vector<stress_test_config>{
        // Original baseline
        {"baseline_1x64x112x112_3x3", {1, 64, 112, 112}, {64, 64, 3, 3}, 200},
        
        // Batch variations (ResNet-like with batch processing)
        {"batch4_1x64x112x112_3x3", {4, 64, 112, 112}, {64, 64, 3, 3}, 150},
        {"batch8_1x64x112x112_3x3", {8, 64, 112, 112}, {64, 64, 3, 3}, 100},
        
        // Odd input dimensions (potential tail/padding issues)
        {"odd_dims_1x64x111x113_3x3", {1, 64, 111, 113}, {64, 64, 3, 3}, 150},
        {"odd_dims_2x64x97x99_3x3", {2, 64, 97, 99}, {64, 64, 3, 3}, 150},
        
        // Very small spatial but many features
        {"small_spatial_1x256x7x7_3x3", {1, 256, 7, 7}, {256, 256, 3, 3}, 150},
        
        // Large spatial (stress tiling & memory access)
        {"large_spatial_1x64x224x224_3x3", {1, 64, 224, 224}, {64, 64, 3, 3}, 100},
        
        // 5x5 kernel (more memory access patterns)
        {"kernel_5x5_1x64x112x112", {1, 64, 112, 112}, {64, 64, 5, 5}, 100},
        
        // Asymmetric spatial dimensions (mismatch tile patterns)
        {"asymmetric_1x64x115x111_3x3", {1, 64, 115, 111}, {64, 64, 3, 3}, 150},
        
        // Feature dimension not multiple of 16 (but supported via padding in fsv16)
        {"features_48_1x48x112x112_3x3", {1, 48, 112, 112}, {48, 48, 3, 3}, 150},
        
        // Output dimensions with non-standard block alignment (e.g., out_x=110 not div by 2/4/8)
        {"output_tail_1x64x116x112_3x3", {1, 64, 116, 112}, {64, 64, 3, 3}, 150},
        
        // Very narrow spatial (emphasize row processing edge cases)
        {"narrow_1x64x5x200_3x3", {1, 64, 5, 200}, {64, 64, 3, 3}, 150},
        {"narrow_1x64x200x5_3x3", {1, 64, 200, 5}, {64, 64, 3, 3}, 150},
        
        // 1x1 convolution stress (different kernel path)
        {"kernel_1x1_1x64x112x112", {1, 64, 112, 112}, {64, 64, 1, 1}, 250},
        
        // Mixed batch + odd dimensions
        {"mixed_1_batch8_97x101_3x3", {8, 64, 97, 101}, {64, 64, 3, 3}, 120},
        
        // Extreme memory pressure: very large tensor (ResNet50 peak)
        // {"extreme_large_1x64x512x512_3x3", {1, 64, 512, 512}, {64, 64, 3, 3}, 50},
        
        // Multiple feature pyramid stages chained (ResNet bottleneck-like)
        {"fpn_stage_4x256x56x56_3x3", {4, 256, 56, 56}, {256, 256, 3, 3}, 80},
        {"fpn_stage_4x512x28x28_1x1", {4, 512, 28, 28}, {512, 512, 1, 1}, 100},
        
        // Stretched memory allocation (max width, min height)
        {"stretched_wide_1x64x2x1024_3x3", {1, 64, 2, 1024}, {64, 64, 3, 3}, 120},
        
        // Very deep feature dimension (ResNet final stages)
        {"deep_features_1x2048x7x7_1x1", {1, 2048, 7, 7}, {2048, 2048, 1, 1}, 60},
    }),
    [](const testing::TestParamInfo<stress_test_config>& info) {
        return info.param.name;
    }
);

// Infrastructure-level stress: network persistence without destruction
// to surface memory leaks or resource exhaustion patterns.
TEST(convolution_gpu_bfyx_f16_infrastructure, network_persistence_memory_reuse) {
    auto& engine = get_test_engine();

    if (!engine.get_device_info().supports_fp16)
        GTEST_SKIP() << "fp16 not supported on this device.";

    const ov::Shape in_shape = {1, 64, 112, 112};
    const ov::Shape wei_shape = {64, 64, 3, 3};
    const int iterations = 500;  // Much longer to stress resource pools

    auto input_mem = engine.allocate_memory({in_shape, data_types::f16, format::bfyx});
    auto weights = engine.allocate_memory({wei_shape, data_types::f16, format::bfyx});

    tests::random_generator rg(GET_SUITE_NAME);
    set_values(input_mem, rg.generate_random_1d<ov::float16>(ov::shape_size(in_shape), -3, 3));
    set_values(weights, rg.generate_random_1d<ov::float16>(ov::shape_size(wei_shape), -3, 3));

    layout dyn_input_layout{ov::PartialShape::dynamic(4), data_types::f16, format::bfyx};
    layout input_fsv16_layout{in_shape, data_types::f16, format::b_fs_yx_fsv16};

    topology topo(
        input_layout("input", dyn_input_layout),
        reorder("input_fsv16", input_info("input"), input_fsv16_layout),
        data("weights", weights),
        convolution("conv", input_info("input_fsv16"), "weights", no_bias, 1,
                    ov::Strides{1, 1}, ov::Strides{1, 1}, ov::CoordinateDiff{0, 0}, ov::CoordinateDiff{0, 0}, false));

    ExecutionConfig cfg = get_test_default_config(engine);
    cfg.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{
        {"conv", {format::b_fs_yx_fsv16, "convolution_gpu_bfyx_f16", impl_types::ocl}}
    }));
    cfg.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    network net(engine, topo, cfg);

    // Keep network alive across many iterations (no destruction between iterations).
    // This simulates real model lifetime and can expose resource pool exhaustion.
    std::string error_msg;
    
    try {
        for (int iter = 0; iter < iterations; iter++) {
            net.set_input_data("input", input_mem);
            auto out = net.execute();
            auto mem = out.at("conv").get_memory();
            cldnn::mem_lock<ov::float16> ptr(mem, get_test_stream());
            EXPECT_GT(ptr.size(), size_t(0));
        }
    } catch (const std::exception& e) {
        error_msg = e.what();
        ADD_FAILURE() << "Network persistence test failed: " << error_msg;
    }
}

// Chained convolution test (ResNet block simulation): tests interaction
// between sequential conv layers without intermediate network destruction.
TEST(convolution_gpu_bfyx_f16_infrastructure, chained_convolutions_resnet_block) {
    auto& engine = get_test_engine();

    if (!engine.get_device_info().supports_fp16)
        GTEST_SKIP() << "fp16 not supported on this device.";

    // Simulate ResNet residual block: in -> conv1 -> conv2 -> add
    // Focus on whether sequential conv execs expose OOB patterns.
    const ov::Shape shape_64x56x56 = {1, 64, 56, 56};
    const ov::Shape shape_64x64_3x3 = {64, 64, 3, 3};
    const int chain_depth = 100;  // 100 sequential block iterations

    auto input_mem = engine.allocate_memory({shape_64x56x56, data_types::f16, format::bfyx});
    auto weights1 = engine.allocate_memory({shape_64x64_3x3, data_types::f16, format::bfyx});
    auto weights2 = engine.allocate_memory({shape_64x64_3x3, data_types::f16, format::bfyx});

    tests::random_generator rg(GET_SUITE_NAME);
    set_values(input_mem, rg.generate_random_1d<ov::float16>(ov::shape_size(shape_64x56x56), -3, 3));
    set_values(weights1, rg.generate_random_1d<ov::float16>(ov::shape_size(shape_64x64_3x3), -3, 3));
    set_values(weights2, rg.generate_random_1d<ov::float16>(ov::shape_size(shape_64x64_3x3), -3, 3));

    layout dyn_input_layout{ov::PartialShape::dynamic(4), data_types::f16, format::bfyx};
    layout input_fsv16_layout{shape_64x56x56, data_types::f16, format::b_fs_yx_fsv16};

    topology topo(
        input_layout("in", dyn_input_layout),
        reorder("in_fsv16", input_info("in"), input_fsv16_layout),
        data("w1", weights1),
        convolution("conv1", input_info("in_fsv16"), "w1", no_bias, 1,
                    ov::Strides{1, 1}, ov::Strides{1, 1}, ov::CoordinateDiff{1, 1}, ov::CoordinateDiff{1, 1}, false),
        data("w2", weights2),
        convolution("conv2", input_info("conv1"), "w2", no_bias, 1,
                    ov::Strides{1, 1}, ov::Strides{1, 1}, ov::CoordinateDiff{1, 1}, ov::CoordinateDiff{1, 1}, false));

    ExecutionConfig cfg = get_test_default_config(engine);
    cfg.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{
        {"conv1", {format::b_fs_yx_fsv16, "convolution_gpu_bfyx_f16", impl_types::ocl}},
        {"conv2", {format::b_fs_yx_fsv16, "convolution_gpu_bfyx_f16", impl_types::ocl}}
    }));
    cfg.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    network net(engine, topo, cfg);

    try {
        for (int iter = 0; iter < chain_depth; iter++) {
            net.set_input_data("in", input_mem);
            auto out = net.execute();
            auto mem = out.at("conv2").get_memory();
            cldnn::mem_lock<ov::float16> ptr(mem, get_test_stream());
            EXPECT_GT(ptr.size(), size_t(0));
        }
    } catch (const std::exception& e) {
        ADD_FAILURE() << "Chained convolution test failed: " << e.what();
    }
}
