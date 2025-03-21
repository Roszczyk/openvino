// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Namespace, property name, default value, [validator], description
OV_CONFIG_RELEASE_OPTION(ov, enable_profiling, false, "Enable profiling for the plugin")
OV_CONFIG_RELEASE_OPTION(ov::device, id, "0", "ID of the current device")
OV_CONFIG_RELEASE_OPTION(ov, cache_dir, "", "Directory where model cache can be stored. Caching is disabled if empty")
OV_CONFIG_RELEASE_OPTION(ov, num_streams, 1, "Number of streams to be used for inference")
OV_CONFIG_RELEASE_OPTION(ov, compilation_num_threads, std::max(1, static_cast<int>(std::thread::hardware_concurrency())), "Max number of CPU threads used for model compilation for the stages that supports parallelism")
OV_CONFIG_RELEASE_OPTION(ov::hint, inference_precision, ov::element::f16, "Model floating-point inference precision. Supported values: { f16, f32, dynamic }", [](ov::element::Type t) { return t == ov::element::f16 || t == ov::element::f32 || t == ov::element::dynamic; })
OV_CONFIG_RELEASE_OPTION(ov::hint, model_priority, ov::hint::Priority::MEDIUM, "High-level hint that defines the priority of the model. It may impact number of threads used for model compilton and inference as well as device queue settings")
OV_CONFIG_RELEASE_OPTION(ov::hint, performance_mode, ov::hint::PerformanceMode::LATENCY, "High-level hint that defines target model inference mode. It may impact number of streams, auto batching, etc")
OV_CONFIG_RELEASE_OPTION(ov::hint, execution_mode, ov::hint::ExecutionMode::PERFORMANCE, "High-level hint that defines the most important metric for the model. Performance mode allows unsafe optimizations that may reduce the model accuracy")
OV_CONFIG_RELEASE_OPTION(ov::hint, num_requests, 0, "Hint that provides number of requests populated by the application")
OV_CONFIG_RELEASE_OPTION(ov::hint, enable_cpu_pinning, false, "Controls if CPU threads are pinned to the cores or not")
OV_CONFIG_RELEASE_OPTION(ov::hint, enable_cpu_reservation, false, "Cpu Reservation means reserve cpus which will not be used by other plugin or compiled model")
OV_CONFIG_RELEASE_OPTION(ov::intel_gpu::hint, host_task_priority, ov::hint::Priority::MEDIUM, "Low-level hint that controls core types used for host tasks")
OV_CONFIG_RELEASE_OPTION(ov::intel_gpu::hint, queue_throttle, ov::intel_gpu::hint::ThrottleLevel::MEDIUM, "Low-level hint that controls the queue throttle level")
OV_CONFIG_RELEASE_OPTION(ov::intel_gpu::hint, queue_priority, ov::hint::Priority::MEDIUM, "Low-level hint that controls queue priority property")
OV_CONFIG_RELEASE_OPTION(ov::intel_gpu::hint, enable_sdpa_optimization, true, "Enable/Disable fused SDPA primitive execution")
OV_CONFIG_RELEASE_OPTION(ov::intel_gpu, enable_loop_unrolling, true, "Enable/Disable Loop/TensorIterator operation unrolling")
OV_CONFIG_RELEASE_OPTION(ov::intel_gpu, disable_winograd_convolution, false, "Enable/Disable winograd convolution implementation if available")
OV_CONFIG_RELEASE_OPTION(ov::internal, exclusive_async_requests, false, "")
OV_CONFIG_RELEASE_OPTION(ov::internal, query_model_ratio, 1.0f, "")
OV_CONFIG_RELEASE_OPTION(ov, cache_mode, ov::CacheMode::OPTIMIZE_SPEED, "Cache mode defines the trade-off between the model compilation time and the disk space required for the cache")
OV_CONFIG_RELEASE_OPTION(ov, cache_encryption_callbacks, ov::EncryptionCallbacks{}, "Callbacks used to encrypt/decrypt the model")
OV_CONFIG_RELEASE_OPTION(ov::hint, dynamic_quantization_group_size, 0, "")
OV_CONFIG_RELEASE_OPTION(ov::hint, kv_cache_precision, ov::element::dynamic, "")
OV_CONFIG_RELEASE_OPTION(ov::intel_gpu::hint, enable_kernels_reuse, false, "")
OV_CONFIG_RELEASE_OPTION(ov, weights_path, "", "Path to the model weights file used for weightless caching")
OV_CONFIG_RELEASE_OPTION(ov::hint, activations_scale_factor, -1.0f, "Scalar floating point value that is used for runtime activation tensor scaling with fp16 inference precision")
OV_CONFIG_RELEASE_OPTION(ov::internal, enable_lp_transformations, false, "Enable/Disable Low precision transformations set")
OV_CONFIG_RELEASE_OPTION(ov::intel_gpu, config_file, "", "Path to custom layers config file")
OV_CONFIG_RELEASE_OPTION(ov::hint, model, nullptr, "Shared pointer to the ov::Model")

OV_CONFIG_RELEASE_INTERNAL_OPTION(ov::intel_gpu, shape_predictor_settings, {10, 16 * 1024, 2, 1.1f}, "Preallocation settings")
OV_CONFIG_RELEASE_INTERNAL_OPTION(ov::intel_gpu, queue_type, QueueTypes::out_of_order, "Type of the queue that must be used for model execution. May be in-order or out-of-order")
OV_CONFIG_RELEASE_INTERNAL_OPTION(ov::intel_gpu, optimize_data, false, "Enable/Disable data flow optimizations for cldnn::program")
OV_CONFIG_RELEASE_INTERNAL_OPTION(ov::intel_gpu, enable_memory_pool, true, "Enable/Disable memory pool usage")
OV_CONFIG_RELEASE_INTERNAL_OPTION(ov::intel_gpu, allow_static_input_reorder, false, "Controls if weights tensors can be reordered during model compilation to more friendly layout for specific kernel")
OV_CONFIG_RELEASE_INTERNAL_OPTION(ov::intel_gpu, custom_outputs, std::vector<std::string>{}, "List of output primitive names")
OV_CONFIG_RELEASE_INTERNAL_OPTION(ov::intel_gpu, force_implementations, ImplForcingMap{}, "Specifies the list of forced implementations for the primitives")
OV_CONFIG_RELEASE_INTERNAL_OPTION(ov::intel_gpu, partial_build_program, false, "Early exit from model compilation process which allows faster execution graph dumping")
OV_CONFIG_RELEASE_INTERNAL_OPTION(ov::intel_gpu, allow_new_shape_infer, false, "Switch between new and old shape inference flow. Shall be removed soon")
OV_CONFIG_RELEASE_INTERNAL_OPTION(ov::intel_gpu, use_onednn, false, "Enable/Disable onednn for usage for particular model/platform")
OV_CONFIG_RELEASE_INTERNAL_OPTION(ov::intel_gpu, use_cm, false, "Enable/Disable CM for usage for particular model/platform")
OV_CONFIG_RELEASE_INTERNAL_OPTION(ov::intel_gpu, max_kernels_per_batch, 8, "Controls how many kernels we combine into batch for more efficient ocl compilation")
OV_CONFIG_RELEASE_INTERNAL_OPTION(ov::intel_gpu, impls_cache_capacity, 300, "Controls capacity of LRU implementations cache that is created for each program object for dynamic models")

OV_CONFIG_DEBUG_GLOBAL_OPTION(ov::intel_gpu, help, false, "Print help message for all config options")
OV_CONFIG_DEBUG_GLOBAL_OPTION(ov::intel_gpu, verbose, 0, "Enable logging for debugging purposes. The higher value the more verbose output. 0 - Disabled, 4 - Maximum verbosity")
OV_CONFIG_DEBUG_GLOBAL_OPTION(ov::intel_gpu, verbose_color, true, "Enable coloring for verbose logs")
OV_CONFIG_DEBUG_GLOBAL_OPTION(ov::intel_gpu, disable_usm, false, "Disable USM memory allocations and use only cl_mem")
OV_CONFIG_DEBUG_GLOBAL_OPTION(ov::intel_gpu, usm_policy, 0, "0: default, 1: use usm_host, 2: do not use usm_host")
OV_CONFIG_DEBUG_GLOBAL_OPTION(ov::intel_gpu, dump_batch_limit, std::numeric_limits<int32_t>::max(), "Max number of batch elements to dump")
OV_CONFIG_DEBUG_GLOBAL_OPTION(ov::intel_gpu, dump_profiling_data_per_iter, false, "Save profiling data w/o per-iteration aggregation")
OV_CONFIG_DEBUG_GLOBAL_OPTION(ov::intel_gpu, log_to_file, "", "Save verbose log to specified file")
OV_CONFIG_DEBUG_GLOBAL_OPTION(ov::intel_gpu, debug_config, "", "Path to debug config in json format")

OV_CONFIG_DEBUG_OPTION(ov::intel_gpu, disable_onednn_post_ops_opt, false, "Disable optimization pass for onednn post-ops")
OV_CONFIG_DEBUG_OPTION(ov::intel_gpu, dump_profiling_data_path, "", "Save csv file with per-stage and per-primitive profiling data to specified folder")
OV_CONFIG_DEBUG_OPTION(ov::intel_gpu, dump_graphs_path, "", "Save intermediate graph representations during model compilation pipeline to specified folder")
OV_CONFIG_DEBUG_OPTION(ov::intel_gpu, dump_sources_path, "", "Save generated sources for each kernel to specified folder")
OV_CONFIG_DEBUG_OPTION(ov::intel_gpu, dump_tensors_path, "", "Save intermediate in/out tensors of each primitive to specified folder")
OV_CONFIG_DEBUG_OPTION(ov::intel_gpu, dump_tensors, ov::intel_gpu::DumpTensors::all, "Tensor types to dump. Supported values: all, in, out")
OV_CONFIG_DEBUG_OPTION(ov::intel_gpu, dump_tensors_format, ov::intel_gpu::DumpFormat::text, "Format of the tensors dump. Supported values: binary, text, text_raw")
OV_CONFIG_DEBUG_OPTION(ov::intel_gpu, dump_layer_names, std::vector<std::string>{}, "Activate dump for specified layers only")
OV_CONFIG_DEBUG_OPTION(ov::intel_gpu, dump_memory_pool_path, "", "Save csv file with memory pool info to specified folder")
OV_CONFIG_DEBUG_OPTION(ov::intel_gpu, dump_memory_pool, false, "Enable verbose output for memory pool")
OV_CONFIG_DEBUG_OPTION(ov::intel_gpu, dump_iterations, std::set<int64_t>{}, "Space separated list of iterations where other dump options should be enabled")
OV_CONFIG_DEBUG_OPTION(ov::intel_gpu, host_time_profiling, 0, "Measre and print host time spent from the beginning of the infer until all host work is done and plugin is ready to block thread on the final clFinish() call")
OV_CONFIG_DEBUG_OPTION(ov::intel_gpu, disable_async_compilation, false, "Disable feature that allows to asyncrhonously prepare static-shaped implementations for the primitives with shape-agnostic kernels selected during compilation")
OV_CONFIG_DEBUG_OPTION(ov::intel_gpu, disable_runtime_buffer_fusing, false, "Disable runtime inplace optimizations for operations like concat and crop")
OV_CONFIG_DEBUG_OPTION(ov::intel_gpu, disable_post_ops_fusions, false, "Disable fusions of operations as post-ops/fused-ops")
OV_CONFIG_DEBUG_OPTION(ov::intel_gpu, disable_horizontal_fc_fusion, false, "Disable pass which merges QKV projections into single MatMul")
OV_CONFIG_DEBUG_OPTION(ov::intel_gpu, disable_fc_swiglu_fusion, false, "Disable pass which merges FC and SwiGLU ops")
OV_CONFIG_DEBUG_OPTION(ov::intel_gpu, disable_fake_alignment, false, "Disable fake alignment feature which tries to keep gpu friendly memory alignment for arbitrary tensor shapes")
OV_CONFIG_DEBUG_OPTION(ov::intel_gpu, disable_memory_reuse, false, "Disable memory reuse for activation tensors")
OV_CONFIG_DEBUG_OPTION(ov::intel_gpu, disable_runtime_skip_reorder, false, "Disable skip reorder optimization applied in runtime")
OV_CONFIG_DEBUG_OPTION(ov::intel_gpu, asym_dynamic_quantization, false, "Enforce asymmetric mode for dynamically quantized activations")
OV_CONFIG_DEBUG_OPTION(ov::intel_gpu, load_dump_raw_binary, std::vector<std::string>{}, "List of layers to load raw binary")
OV_CONFIG_DEBUG_OPTION(ov::intel_gpu, start_after_processes, std::vector<std::string>{}, "Start inference after specified list of processes")
OV_CONFIG_DEBUG_OPTION(ov::intel_gpu, dry_run_path, "", "Enables mode which partially compiles a model and stores runtime model into specified directory")
