// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ze_common.hpp"
#include "ze_module_holder.hpp"

#include <memory>

namespace cldnn {
namespace ze {

// RAII wrapper for Level Zero kernel
class ze_kernel_holder {
public:
    // Take ownership of existing kernel handle
    explicit ze_kernel_holder(ze_kernel_handle_t kernel, std::shared_ptr<ze_module_holder> module) : m_kernel(kernel), m_module(module) {
        std::cout << "Kernel holder created " << std::endl;
    }
    ze_kernel_holder(const ze_kernel_holder& other) = delete;
    ze_kernel_holder& operator=(const ze_kernel_holder& other) = delete;
    ~ze_kernel_holder() {
        ze_module_handle_t module_handle = m_module->get_module_handle();
        uint32_t kernel_count = 0;
        zeModuleGetKernelNames(module_handle, &kernel_count, nullptr);
        std::vector<const char*> kernel_names(kernel_count);
        // Specification does not mention who is responsible for the returned pointers
        // Assume Level Zero owns the pointers and they will remain valid as long as the module resource
        zeModuleGetKernelNames(module_handle, &kernel_count, kernel_names.data());
        std::cout << "Kernel destroyed: ";
        for (auto name : kernel_names) {
            std::cout << name << " ";
        }
        std::cout << std::endl;
        OV_ZE_WARN(zeKernelDestroy(m_kernel));
    }
    ze_kernel_handle_t get_kernel_handle() { return m_kernel; }
    std::shared_ptr<ze_module_holder> get_module() { return m_module; }
private:
    ze_kernel_handle_t m_kernel;
    std::shared_ptr<ze_module_holder> m_module;
};
}  // namespace ze
}  // namespace cldnn
