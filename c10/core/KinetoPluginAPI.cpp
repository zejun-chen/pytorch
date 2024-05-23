#ifdef USE_KINETO

#include "KinetoPluginAPI.h"

#include <map>
#include <vector>
#include <mutex>

namespace c10 {
namespace kineto_plugin {

static std::map<DeviceType, std::vector<PluginFactory>> plugin_factories;
static std::mutex mutex;

void insertKinetoPluginRegister(DeviceType t, PluginFactory f) {
  std::scoped_lock<std::mutex> lock(mutex);
  auto it = plugin_factories.find(t);
  if (it != plugin_factories.end()) {
    it->second.push_back(f);
  } else {
    plugin_factories[t] = std::vector<PluginFactory>{f};
  }
}

void removeKinetoPluginRegister(DeviceType t) {
  std::scoped_lock<std::mutex> lock(mutex);
  auto it = plugin_factories.find(t);
  if (it != plugin_factories.end())
    plugin_factories.erase(it);
}

void registerKinetoPluginProfiler() {
  for (const auto& pair : plugin_factories) {
    const auto& factories = pair.second;
    for (const auto& factory : factories)
      libkineto::api().registerProfilerFactory(factory);
  }
}

void registerKinetoPluginProfiler(DeviceType t) {
  auto it = plugin_factories.find(t);
  if (it != plugin_factories.end()) {
    const auto& factories = it->second;
    for (const auto& factory : factories)
      libkineto::api().registerProfilerFactory(factory);
  }
}

} // namespace kineto_plugin
} // namespace c10

#endif
