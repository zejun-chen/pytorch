#pragma once

#ifdef USE_KINETO

#include <c10/core/DeviceType.h>
#include <c10/macros/Macros.h>

#include <libkineto.h>

namespace c10 {
namespace kineto_plugin {

using PluginFactory = libkineto::ChildActivityProfilerFactory;
void C10_API insertKinetoPluginRegister(DeviceType t, PluginFactory f);
void C10_API registerKinetoPluginProfiler();

struct KinetoPluginRegister {
  explicit KinetoPluginRegister(DeviceType t, PluginFactory f) {
    c10::kineto_plugin::insertKinetoPluginRegister(t, f);
  }
};

#define REGISTER_KINETO_PLUGIN(t, f)                        \
  namespace {                                               \
  static KinetoPluginRegister kineto_plugin_register(t, f); \
  }

} // namespace kineto_plugin
} // namespace c10

#endif
