#pragma once

#include <libkineto.h>
#include <output_base.h>
#include <time_since_epoch.h>

#include <pti/pti_view.h>
#include <sycl/sycl.hpp>

namespace at::kineto_plugin::xpu {

#define XPU_PTI_CHECK(returnCode)                                             \
  {                                                                           \
    if (returnCode != PTI_SUCCESS) {                                          \
      std::string funcMsg(__func__);                                          \
      std::string codeMsg = std::to_string(returnCode);                       \
      std::string HeadMsg("Kineto Profiler on XPU got error from function "); \
      std::string Msg(". The error code is ");                                \
      throw std::runtime_error(HeadMsg + funcMsg + Msg + codeMsg);            \
    }                                                                         \
  }

class XPUActivityApi;

using act_t = libkineto::ActivityType;
using logger_t = libkineto::ActivityLogger;
using itrace_t = libkineto::ITraceActivity;
using gtrace_t = libkineto::GenericTraceActivity;
using DeviceIndex_t = int8_t;

} // namespace at::kineto_plugin::xpu
