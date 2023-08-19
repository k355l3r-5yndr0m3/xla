#include "plugin.h"
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

#include "xla/pjrt/pjrt_api.h"

// #include "xla/pjrt/c/pjrt_c_api.h"
// #include "xla/pjrt/c/pjrt_c_api_helpers.h"
// #include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
// 
// #include "xla/pjrt/local_device_state.h"
// #include "xla/pjrt/pjrt_client.h"
// #include "xla/pjrt/pjrt_executable.h"
// #include "xla/pjrt/pjrt_stream_executor_client.h"
// 
// #include "xla/client/client_library.h"
// #include "xla/service/platform_util.h"
// #include "xla/stream_executor/platform.h"
// #include "xla/stream_executor/stream_executor_pimpl.h"


const PJRT_Api* GetPjrtApi() { return pjrt::PjrtApi("CPU").value_or(nullptr); }


