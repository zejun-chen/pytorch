#include <c10/util/CallOnce.h>
#include <c10/util/Exception.h>
#include <c10/xpu/XPUFunctions.h>

#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace c10::xpu {
namespace {

/*
 * Note [Device Management]
 *
 * An Intel GPU device qualifies as a type of SYCL device. This classification
 * allows for the runtime querying of Intel GPU device information through the
 * SYCL runtime library.
 *
 * Device status is managed through a SYCL device pool, with SYCL devices
 * determined at runtime. There's currently a SYCL device pool that is lazily
 * created and only initialized once, ensuring thread-local safety. Each device
 * within the device pool shares the same default context.
 */
c10::once_flag init_flag;
thread_local DeviceIndex curDeviceIndex = 0;

struct DevicePool {
  std::vector<std::unique_ptr<sycl::device>> devices;
  std::vector<std::array<unsigned char, 16>> uuids;
  std::unique_ptr<sycl::context> context;
} gDevicePool;

void enumDevices() {
  auto platform_list = sycl::platform::get_platforms();
  // Enumerated GPU devices from the specific platform.
  for (const auto& platform : platform_list) {
    if (platform.get_backend() != sycl::backend::ext_oneapi_level_zero) {
      continue;
    }
    auto device_list = platform.get_devices();
    for (const auto& device : device_list) {
      if (device.is_gpu()) {
        gDevicePool.devices.push_back(std::make_unique<sycl::device>(device));
        if (device.has(sycl::aspect::ext_intel_device_info_uuid))
          gDevicePool.uuids.push_back(
              device.get_info<sycl::ext::intel::info::device::uuid>());
        else
          gDevicePool.uuids.push_back(std::array<unsigned char, 16>{});
      }
    }
  }
}

inline void initGlobalDevicePoolState() {
  // Enumerate all GPU devices and record them.
  enumDevices();
  if (gDevicePool.devices.empty()) {
    TORCH_WARN("XPU device count is zero!");
    return;
  }

  // The default context is utilized for each Intel GPU device, allowing the
  // retrieval of the context from any GPU device.
  gDevicePool.context = std::make_unique<sycl::context>(
      gDevicePool.devices[0]->get_platform().ext_oneapi_get_default_context());
}

inline void initDevicePoolCallOnce() {
  c10::call_once(init_flag, initGlobalDevicePoolState);
}

void initDeviceProperties(DeviceProp* device_prop, int device) {
  using namespace sycl::info;
  using namespace sycl::ext;
  // Get raw sycl device associated with device index.
  auto& raw_device = *gDevicePool.devices[device];

  // Initialize the device properties associated with the specific device.
#define ASSIGN_DEVICE_PROP(property) \
  device_prop->property = raw_device.get_info<device::property>();

#define ASSIGN_EXT_DEVICE_PROP(property, default_value)                      \
  device_prop->property = raw_device.has(sycl::aspect::ext_intel_##property) \
      ? raw_device.get_info<intel::info::device::property>()                 \
      : default_value;

#define ASSIGN_DEVICE_ASPECT(member) \
  device_prop->has_##member = raw_device.has(sycl::aspect::member);

  AT_FORALL_XPU_DEVICE_PROPERTIES(ASSIGN_DEVICE_PROP);

  device_prop->platform_name =
      raw_device.get_info<device::platform>().get_info<platform::name>();

  AT_FORALL_XPU_EXT_DEVICE_PROPERTIES(ASSIGN_EXT_DEVICE_PROP);

  AT_FORALL_XPU_DEVICE_ASPECT(ASSIGN_DEVICE_ASPECT);
  return;
}

inline void check_device(DeviceIndex device) {
  // TODO: Use c10::Device::MAX_NUM_DEVICES directly. DeviceIndex is a int8_t
  // value, and the maximum number of GPUs that PyTorch recognizes is 64. So, we
  // have to check if there is an overflow happen. When DeviceIndex changes to
  // int16_t and c10::Device::MAX_NUM_DEVICES is provided, we should use it
  // directly to check if too many XPU devices are detected.
  TORCH_CHECK(
      gDevicePool.devices.size() <= std::numeric_limits<DeviceIndex>::max(),
      "Too many XPU devices, DeviceIndex overflowed");
  auto total = static_cast<DeviceIndex>(gDevicePool.devices.size());
  TORCH_CHECK(
      device >= 0 && device < total,
      "device is out of range, device is ",
      device,
      ", total number of device is ",
      total,
      ".");
}

} // anonymous namespace

sycl::device& get_raw_device(DeviceIndex device) {
  initDevicePoolCallOnce();
  check_device(device);
  return *gDevicePool.devices[device];
}

sycl::context& get_device_context() {
  initDevicePoolCallOnce();
  TORCH_CHECK(
      gDevicePool.context,
      "Device pool initialization failed, you might not have an XPU device.")
  return *gDevicePool.context;
}

void get_device_properties(DeviceProp* device_prop, DeviceIndex device) {
  initDevicePoolCallOnce();
  TORCH_CHECK(device_prop, "device_prop is an invalid pointer.");
  check_device(device);
  initDeviceProperties(device_prop, device);
}

DeviceIndex get_device_idx_from_pointer(void* ptr) {
  initDevicePoolCallOnce();
  TORCH_CHECK(ptr, "ptr is an invalid pointer.");
  auto type = sycl::get_pointer_type(ptr, get_device_context());
  TORCH_CHECK(
      type == sycl::usm::alloc::device, "ptr is not a device type pointer.");

  sycl::device raw_device = sycl::get_pointer_device(ptr, get_device_context());
  auto match_device = [raw_device](const auto& device) -> bool {
    return raw_device == *device;
  };
  auto it = std::find_if(
      gDevicePool.devices.begin(), gDevicePool.devices.end(), match_device);
  TORCH_CHECK(
      it != gDevicePool.devices.end(),
      "Can't find the pointer from XPU devices.");
  return static_cast<DeviceIndex>(
      std::distance(gDevicePool.devices.begin(), it));
}

DeviceIndex get_device_idx_from_uuid(const uint8_t device_uuid[16]) {
  std::array<unsigned char, 16> key;
  memcpy(key.data(), device_uuid, 16);
  auto it = std::find(gDevicePool.uuids.begin(), gDevicePool.uuids.end(), key);
  if (it != gDevicePool.uuids.end())
    return static_cast<DeviceIndex>(
        std::distance(gDevicePool.uuids.begin(), it));
  return static_cast<DeviceIndex>(-1);
}

DeviceIndex device_count() {
  initDevicePoolCallOnce();
  return static_cast<DeviceIndex>(gDevicePool.devices.size());
}

DeviceIndex device_count_ensure_non_zero() {
  auto count = device_count();
  // Zero gpus could produce a warning in `device_count` but we fail here.
  TORCH_CHECK(count, "No XPU devices are available.");
  return count;
}

DeviceIndex current_device() {
  initDevicePoolCallOnce();
  return curDeviceIndex;
}

void set_device(DeviceIndex device) {
  initDevicePoolCallOnce();
  check_device(device);
  curDeviceIndex = device;
}

c10::DeviceIndex exchange_device(c10::DeviceIndex to_device) {
  auto cur_device = current_device();
  if (to_device == cur_device) {
    return cur_device;
  }
  set_device(to_device);
  return cur_device;
}

c10::DeviceIndex maybe_exchange_device(c10::DeviceIndex to_device) {
  return exchange_device(to_device);
}

} // namespace c10::xpu
