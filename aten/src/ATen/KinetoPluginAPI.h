#pragma once

#include <c10/core/DeviceType.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>

#include <libkineto.h>

namespace at::kineto_plugin {

using PluginFactory = libkineto::ChildActivityProfilerFactory;
TORCH_API void SetKinetoPluginRegister(
    DeviceType t,
    PluginFactory func,
    uint8_t priority = 0);
TORCH_API PluginFactory GetKinetoPluginRegister(const DeviceType& t);

void registerKinetoPluginProfiler();

template <DeviceType t>
struct KinetoPluginRegisterer {
  explicit KinetoPluginRegisterer(PluginFactory func) {
    SetKinetoPluginRegister(t, func);
  }
};

#define REGISTER_KINETO_PLUGIN(t, f)                                          \
  namespace {                                                                 \
  static at::kineto_plugin::KinetoPluginRegisterer<t> g_kineto_profiler_d(f); \
  }

} // namespace at::kineto_plugin
