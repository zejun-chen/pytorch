#pragma once

#include <libkineto.h>
#include <output_base.h>
#include <time_since_epoch.h>

#include <pti/pti_view.h>

#include <c10/util/Exception.h>
#include <c10/xpu/XPUMacros.h>

namespace at::kineto_plugin::xpu {

#define AT_XPU_PTI_CHECK(returnCode)                       \
  {                                                        \
    TORCH_CHECK(                                           \
        returnCode == PTI_SUCCESS,                         \
        "Kineto Profiler on XPU got error from function ", \
        __func__,                                          \
        ". The error code is ",                            \
        returnCode);                                       \
  }

class XPUActivityApi;

using act_t = libkineto::ActivityType;
using logger_t = libkineto::ActivityLogger;
using itrace_t = libkineto::ITraceActivity;
using gtrace_t = libkineto::GenericTraceActivity;

} // namespace at::kineto_plugin::xpu
