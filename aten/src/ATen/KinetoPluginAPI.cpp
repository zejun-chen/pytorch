#include <c10/core/DeviceType.h>
#include <ATen/KinetoPluginAPI.h>

#include <map>
#include <vector>
#include <mutex>

namespace at::kineto_plugin {

static std::map<DeviceType, PluginFactory> plugin_factories;
static std::map<DeviceType, uint8_t> plugin_priorities;
static std::mutex mutex;

void SetKinetoPluginRegister(DeviceType t, PluginFactory func, uint8_t priority) {
  std::scoped_lock<std::mutex> lock(mutex);
  auto priority_it = plugin_priorities.find(t);
  if (priority_it != plugin_priorities.end()) {
    if (priority >= plugin_priorities[t]) {
      plugin_factories[t] = func;
      plugin_priorities[t] = priority;
    }
  } else {
    plugin_factories[t] = func;
    plugin_priorities[t] = priority;
  }
}

PluginFactory GetKinetoPluginRegister(const DeviceType& t) {
  std::scoped_lock<std::mutex> lock(mutex);
  auto it = plugin_factories.find(t);
  if (it != plugin_factories.end()) {
    return plugin_factories[t];
  } else {
    std::string msg = "No kineto profiler register function is found for the device " + DeviceTypeName(t, true);
    TORCH_CHECK(false, msg);
  }
}

void registerKinetoPluginProfiler() {
  for (const auto& pair : plugin_factories) {
    const auto& factory = pair.second;
    libkineto::api().registerProfilerFactory(factory);
  }
}

} // namespace at::kineto_plugin
