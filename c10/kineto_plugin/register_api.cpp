#include "register_api.h"

#ifdef USE_XPU
#include <c10/kineto_plugin/xpu/XPUActivityProfiler.h>
#endif

namespace c10 {
namespace kineto_plugin {

#define REGISTER_KINETO_PLUGIN(TYPE)                          \
  #ifdef USE_##TYPE                                           \
  libkineto::api().registerProfilerFactory(                   \
      []() -> std::unique_ptr<IActivityProfiler> {            \
        return std::make_unique<TYPE##ActivityProfiler>(); \
      });                                                     \
  #endif

// register new kineto backends in kineto plugin
void registerKinetoPluginProfiler() {
  REGISTER_KINETO_PLUGIN(XPU);
}

}
}
