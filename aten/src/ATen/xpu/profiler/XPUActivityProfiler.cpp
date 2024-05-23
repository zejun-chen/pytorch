#include <ATen/xpu/profiler/XPUActivityApi.h>
#include <ATen/xpu/profiler/XPUActivityProfiler.h>

#include <ATen/KinetoPluginAPI.h>

#include <chrono>

namespace at::kineto_plugin::xpu {

uint32_t XPUActivityProfilerSession::iterationCount_ = 0;
std::vector<std::array<unsigned char, 16>>
    XPUActivityProfilerSession::deviceUUIDs_ = {};

// =========== Session Constructor ============= //
XPUActivityProfilerSession::XPUActivityProfilerSession(
    XPUActivityApi& xpti,
    const libkineto::Config& config,
    const std::set<act_t>& activity_types)
    : xpti_(xpti), config_(config.clone()), activity_types_(activity_types) {
  xpti_.setMaxBufferSize(config_->activitiesMaxGpuBufferSize());
  xpti_.enablePtiActivities(activity_types_);
  enumDeviceUUIDs();
}

XPUActivityProfilerSession::~XPUActivityProfilerSession() {
  xpti_.clearActivities();
}

// =========== Session Public Methods ============= //
void XPUActivityProfilerSession::start() {
  profilerStartTs_ =
      libkineto::timeSinceEpoch(std::chrono::high_resolution_clock::now());
}

void XPUActivityProfilerSession::stop() {
  xpti_.disablePtiActivities(activity_types_);
  profilerEndTs_ =
      libkineto::timeSinceEpoch(std::chrono::high_resolution_clock::now());
}

void XPUActivityProfilerSession::processTrace(logger_t& logger) {
  traceBuffer_.span = libkineto::TraceSpan(
      profilerStartTs_, profilerEndTs_, "__xpu_profiler__");
  traceBuffer_.span.iteration = iterationCount_++;
  auto gpuBuffer = xpti_.activityBuffers();
  if (gpuBuffer) {
    xpti_.processActivities(
        *gpuBuffer,
        std::bind(
            &XPUActivityProfilerSession::handlePtiActivity,
            this,
            std::placeholders::_1,
            &logger));
  }
}

void XPUActivityProfilerSession::processTrace(
    logger_t& logger,
    libkineto::getLinkedActivityCallback get_linked_activity,
    int64_t captureWindowStartTime,
    int64_t captureWindowEndTime) {
  captureWindowStartTime_ = captureWindowStartTime;
  captureWindowEndTime_ = captureWindowEndTime;
  cpuActivity_ = get_linked_activity;
  processTrace(logger);
}

std::unique_ptr<libkineto::DeviceInfo> XPUActivityProfilerSession::
    getDeviceInfo() {
  return {};
}

std::vector<libkineto::ResourceInfo> XPUActivityProfilerSession::
    getResourceInfos() {
  return {};
}

std::unique_ptr<libkineto::CpuTraceBuffer> XPUActivityProfilerSession::
    getTraceBuffer() {
  return std::make_unique<libkineto::CpuTraceBuffer>(std::move(traceBuffer_));
}

void XPUActivityProfilerSession::pushCorrelationId(uint64_t id) {
  xpti_.pushCorrelationID(id, XPUActivityApi::CorrelationFlowType::Default);
}

void XPUActivityProfilerSession::popCorrelationId() {
  xpti_.popCorrelationID(XPUActivityApi::CorrelationFlowType::Default);
}

void XPUActivityProfilerSession::pushUserCorrelationId(uint64_t id) {
  xpti_.pushCorrelationID(id, XPUActivityApi::CorrelationFlowType::User);
}

void XPUActivityProfilerSession::popUserCorrelationId() {
  xpti_.popCorrelationID(XPUActivityApi::CorrelationFlowType::User);
}

void XPUActivityProfilerSession::enumDeviceUUIDs() {
  if (!deviceUUIDs_.empty()) {
    return;
  }
  auto platform_list = sycl::platform::get_platforms();
  // Enumerated GPU devices from the specific platform.
  for (const auto& platform : platform_list) {
    if (platform.get_backend() != sycl::backend::ext_oneapi_level_zero) {
      continue;
    }
    auto device_list = platform.get_devices();
    for (const auto& device : device_list) {
      if (device.is_gpu()) {
        if (device.has(sycl::aspect::ext_intel_device_info_uuid)) {
          deviceUUIDs_.push_back(
              device.get_info<sycl::ext::intel::info::device::uuid>());
        } else {
          std::cerr
              << "Warnings: UUID is not supported for this XPU device. The device index of records will be 0."
              << std::endl;
          deviceUUIDs_.push_back(std::array<unsigned char, 16>{});
        }
      }
    }
  }
}

DeviceIndex_t XPUActivityProfilerSession::getDeviceIdxFromUUID(
    const uint8_t deviceUUID[16]) {
  std::array<unsigned char, 16> key;
  memcpy(key.data(), deviceUUID, 16);
  auto it = std::find(deviceUUIDs_.begin(), deviceUUIDs_.end(), key);
  if (it == deviceUUIDs_.end()) {
    std::cerr
        << "Warnings: Can't find the legal XPU device from the given UUID."
        << std::endl;
    return static_cast<DeviceIndex_t>(0);
  }
  return static_cast<DeviceIndex_t>(std::distance(deviceUUIDs_.begin(), it));
}

// =========== ActivityProfiler Public Methods ============= //
// FIXME: remove here legacy data struct
const std::set<act_t> kXpuTypes{
    act_t::GPU_MEMCPY,
    act_t::GPU_MEMSET,
    act_t::CONCURRENT_KERNEL,
    act_t::XPU_RUNTIME,
    // act_t::EXTERNAL_CORRELATION,
    // act_t::OVERHEAD,
};

const std::string& XPUActivityProfiler::name() const {
  return name_;
}

const std::set<act_t>& XPUActivityProfiler::availableActivities() const {
  // FIXME: Remove the legacy member function availableActivities of
  // IActivityProfiler, then the derived class can remove the override version.
  throw std::runtime_error(
      "The availableActivities is legacy method and should not be called by kineto");
  return kXpuTypes;
}

std::unique_ptr<libkineto::IActivityProfilerSession> XPUActivityProfiler::
    configure(
        const std::set<act_t>& activity_types,
        const libkineto::Config& config) {
  return std::make_unique<XPUActivityProfilerSession>(
      XPUActivityApi::singleton(), config, activity_types);
}

std::unique_ptr<libkineto::IActivityProfilerSession> XPUActivityProfiler::
    configure(
        int64_t ts_ms,
        int64_t duration_ms,
        const std::set<act_t>& activity_types,
        const libkineto::Config& config) {
  AsyncProfileStartTime_ = ts_ms;
  AsyncProfileEndTime_ = ts_ms + duration_ms;
  return configure(activity_types, config);
}

// Global register kineto plugin once at first.
auto XPUPluginRegister = []() -> std::unique_ptr<libkineto::IActivityProfiler> {
  auto returnCode = ptiViewGPULocalAvailable();
  if (returnCode != PTI_SUCCESS) {
    std::string errCode = std::to_string(returnCode);
    std::string errMsg(
        "Fail to enable Kineto Profiler on XPU due to error code: ");
    throw std::runtime_error(errMsg + errCode);
  }
  return std::make_unique<XPUActivityProfiler>();
};
REGISTER_KINETO_PLUGIN(kXPU, XPUPluginRegister);

} // namespace at::kineto_plugin::xpu
