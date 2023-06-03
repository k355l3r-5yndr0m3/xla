#include "plugin.h"
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"

#include "xla/pjrt/local_device_state.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_stream_executor_client.h"

#include "xla/client/client_library.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor_pimpl.h"

namespace pjrt::plugin {

PJRT_Error* PJRT_Client_Create(PJRT_Client_Create_Args* args) {
    PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
        "PJRT_Client_Create_Args", PJRT_Client_Create_Args_STRUCT_SIZE,
        args->struct_size));

    xla::LocalClient* local_client = xla::ClientLibrary::LocalClientOrDie();

    PJRT_ASSIGN_OR_RETURN(
        xla::se::Platform *platform,
        xla::PlatformUtil::GetDefaultPlatform());

    xla::se::StreamExecutorConfig config;
    config.ordinal = 0;

    PJRT_ASSIGN_OR_RETURN(
        xla::se::StreamExecutor *executor,
        platform->GetExecutor(config));

    auto device_state = std::make_unique<xla::LocalDeviceState>(
        executor, local_client, xla::LocalDeviceState::kSynchronous,
        /*max_inflight_computations=*/32,
        /*allow_event_reuse=*/false, /*use_callback_stream=*/false);

    auto device = std::make_unique<xla::PjRtStreamExecutorDevice>(
        0, std::move(device_state), platform->Name());

    std::vector<std::unique_ptr<xla::PjRtStreamExecutorDevice>> devices;
    devices.emplace_back(std::move(device));

    auto client = std::make_unique<xla::PjRtStreamExecutorClient>(
        platform->Name(), local_client, std::move(devices), /*process_index=*/0,
        /*allocator=*/nullptr, /*host_memory_allocator=*/nullptr,
        /*should_stage_host_to_device_transfers=*/false,
        /*gpu_run_options=*/nullptr);

    args->client = CreateWrapperClient(std::move(client));
    return nullptr;
}

PJRT_Error* PJRT_CpuDeviceTopology_Create(PJRT_TopologyDescription_Create_Args* args) {
    return new PJRT_Error{tsl::errors::Unimplemented(
        "Topology not supported for CPU compilation.")};
}


} // namespace pjrt::plugin

constexpr PJRT_Api pjrt_api =
    pjrt::CreatePjrtApi(pjrt::plugin::PJRT_Client_Create,
                        pjrt::plugin::PJRT_CpuDeviceTopology_Create);

const PJRT_Api* GetPjrtApi() { return &pjrt_api; }


const void *SerializeCompileOptions(size_t *compile_options_size) {
    xla::CompileOptions options;
    
    auto result = options.ToProto();
    if (!result.ok())
        return NULL;

    auto proto = result.value();
    size_t size = proto.ByteSizeLong();
    void *buffer = malloc(size);
    if (!proto.SerializeToArray(buffer, size)) {
        free(buffer);
        return NULL;
    }
    
    *compile_options_size = size;
    return buffer;
}
