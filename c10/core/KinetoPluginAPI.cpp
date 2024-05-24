#ifdef USE_KINETO

#include "KinetoPluginAPI.h"

#include <map>
#include <vector>
#include <mutex>

namespace c10 {
namespace kineto_plugin {

static std::map<DeviceType, PluginFactory> plugin_factories;
static std::mutex mutex;

void insertKinetoPluginRegister(DeviceType t, PluginFactory f) {
  std::scoped_lock<std::mutex> lock(mutex);
  auto it = plugin_factories.find(t);
  if (it != plugin_factories.end()) {
    it->second = f;
  } else {
    plugin_factories[t] = f;
  }
}

void registerKinetoPluginProfiler() {
  for (const auto& pair : plugin_factories) {
    const auto& factory = pair.second;
    libkineto::api().registerProfilerFactory(factory);
}


} // namespace kineto_plugin
} // namespace c10

#endif
