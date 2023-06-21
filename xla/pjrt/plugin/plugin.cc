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

// To fix the error where CompileOptions is not accessible 
// Because the virtual method GetCompileOptions is not overriden any where
static std::vector<std::vector<xla::PjRtBuffer*>> Convert2DCBuffersToCppBuffers(
    PJRT_Buffer*** c_lists, size_t outer_size, size_t inner_size) {
  std::vector<std::vector<xla::PjRtBuffer*>> cpp_lists;
  cpp_lists.reserve(outer_size);
  for (int i = 0; i < outer_size; ++i) {
    auto& cpp_list = cpp_lists.emplace_back();
    cpp_list.reserve(inner_size);
    for (int j = 0; j < inner_size; ++j) {
      cpp_list.push_back(c_lists[i][j]->buffer.get());
    }
  }
  return cpp_lists;
}

static xla::SendCallback CSendCallbackToCpp(
    const PJRT_SendCallbackInfo& c_callback) {
  return xla::SendCallback{
      c_callback.channel_id,
      // Transfer metadata is unused because PJRT C API doesn't support
      // use_major_to_minor_data_layout_for_callbacks = false
      [user_arg = c_callback.user_arg, callback = c_callback.send_callback](
          const xla::PjRtTransferMetadata& unused_metadata,
          xla::PjRtChunk input, size_t total_size_in_bytes,
          bool done) -> xla::Status {
        PJRT_Chunk c_chunk = ConvertFromCppChunk(std::move(input));
        // PJRT_CallbackError creates PJRT_Error in the implementation, but
        // using the caller's callback status code & message. This way, the
        // caller avoids creating PJRT_Error itself, and the PJRT_Error is fully
        // managed in the implementation layer.
        PJRT_CallbackError c_callback_error =
            [](PJRT_Error_Code code, const char* message, size_t message_size) {
              return new PJRT_Error{
                  xla::Status(static_cast<absl::StatusCode>(code),
                              std::string(message, message_size))};
            };

        std::unique_ptr<PJRT_Error> error(callback(
            &c_chunk, &c_callback_error, total_size_in_bytes, done, user_arg));
        if (error == nullptr) {
          return tsl::OkStatus();
        }
        return error->status;
      }};
}

static xla::RecvCallback CRecvCallbackToCpp(
    const PJRT_RecvCallbackInfo& c_callback) {
  return xla::RecvCallback{
      c_callback.channel_id,
      // Transfer metadata is unused because PJRT C API doesn't support
      // use_major_to_minor_data_layout_for_callbacks = false
      [user_arg = c_callback.user_arg, callback = c_callback.recv_callback](
          const xla::PjRtTransferMetadata& unused_metadata,
          std::unique_ptr<xla::CopyToDeviceStream> stream) {
        PJRT_CopyToDeviceStream c_stream{std::move(stream)};
        callback(&c_stream, user_arg);
      }};
}

static void CRecvCallbackListsToCpp(
    PJRT_RecvCallbackInfo** c_lists, size_t outer_size, size_t inner_size,
    std::vector<std::vector<xla::RecvCallback>>& cpp_lists) {
  cpp_lists.reserve(outer_size);
  for (int i = 0; i < outer_size; ++i) {
    auto& cpp_list = cpp_lists.emplace_back();
    cpp_list.reserve(inner_size);
    for (int j = 0; j < inner_size; ++j) {
      cpp_list.push_back(CRecvCallbackToCpp(c_lists[i][j]));
    }
  }
}

static void CSendCallbackListsToCpp(
    PJRT_SendCallbackInfo** c_lists, size_t outer_size, size_t inner_size,
    std::vector<std::vector<xla::SendCallback>>& cpp_lists) {
  cpp_lists.reserve(outer_size);
  for (int i = 0; i < outer_size; ++i) {
    std::vector<xla::SendCallback>& cpp_list = cpp_lists.emplace_back();
    cpp_list.reserve(inner_size);
    for (int j = 0; j < inner_size; ++j) {
      cpp_list.push_back(CSendCallbackToCpp(c_lists[i][j]));
    }
  }
}

PJRT_Error* PJRT_LoadedExecutable_Execute(
    PJRT_LoadedExecutable_Execute_Args* args) {
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes(
      "PJRT_LoadedExecutable_Execute_Args",
      PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE, args->struct_size));
  PJRT_RETURN_IF_ERROR(CheckMatchingStructSizes("PJRT_ExecuteOptions",
                                                PJRT_ExecuteOptions_STRUCT_SIZE,
                                                args->options->struct_size));

  xla::ExecuteOptions options;
  options.launch_id = args->options->launch_id;
  options.strict_shape_checking = true;
  options.arguments_are_tupled = false;
  options.untuple_result = true;
  options.context = nullptr;
  options.multi_slice_config = nullptr;
  options.use_major_to_minor_data_layout_for_callbacks = true;

  std::vector<std::vector<xla::PjRtBuffer*>> cpp_argument_lists =
      Convert2DCBuffersToCppBuffers(args->argument_lists, args->num_devices,
                                    args->num_args);

  // Set send/recv callbacks in ExecuteOptions. The callbacks
  // should call the C callbacks provided by the caller.
  auto cpp_send_callbacks =
      std::make_shared<std::vector<std::vector<xla::SendCallback>>>();
  if (args->options->num_send_ops > 0) {
    CSendCallbackListsToCpp(args->options->send_callbacks, args->num_devices,
                            args->options->num_send_ops, *cpp_send_callbacks);
    options.send_callbacks = *cpp_send_callbacks;
    CHECK_EQ(options.send_callbacks.size(), args->num_devices);
  }

  auto cpp_recv_callbacks =
      std::make_shared<std::vector<std::vector<xla::RecvCallback>>>();
  if (args->options->num_recv_ops > 0) {
    CRecvCallbackListsToCpp(args->options->recv_callbacks, args->num_devices,
                            args->options->num_recv_ops, *cpp_recv_callbacks);
    options.recv_callbacks = *cpp_recv_callbacks;
    CHECK_EQ(options.recv_callbacks.size(), args->num_devices);
  }

  if (args->execute_device == nullptr) {
    std::vector<std::vector<std::unique_ptr<xla::PjRtBuffer>>> cpp_buffer_lists;
    if (args->device_complete_events != nullptr ||
        !cpp_send_callbacks->empty() || !cpp_recv_callbacks->empty()) {
      std::optional<std::vector<xla::PjRtFuture<xla::Status>>> returned_futures;
      returned_futures.emplace();
      PJRT_ASSIGN_OR_RETURN(cpp_buffer_lists,
                            args->executable->get()->Execute(
                                cpp_argument_lists, options, returned_futures));
      CHECK_EQ(returned_futures->size(), args->num_devices);

      // We assume that these OnReady callbacks will fire even if
      // returned_futures is destroyed first. This is true for the
      // AsyncValue-based implementation of PjRtFuture.
      if (!cpp_send_callbacks->empty() || !cpp_recv_callbacks->empty()) {
        for (int i = 0; i < returned_futures->size(); ++i) {
          (*returned_futures)[i].OnReady(
              [cpp_send_callbacks, cpp_recv_callbacks](xla::Status status) {
                // Keeps C++ callbacks alive until execution completes on all
                // devices.
              });
        }
      }

      if (args->device_complete_events != nullptr) {
        for (int i = 0; i < returned_futures->size(); ++i) {
          args->device_complete_events[i] =
              new PJRT_Event{std::move((*returned_futures)[i])};
        }
      }
    } else {
      PJRT_ASSIGN_OR_RETURN(cpp_buffer_lists, args->executable->get()->Execute(
                                                  cpp_argument_lists, options));
    }

    for (int i = 0; i < cpp_buffer_lists.size(); ++i) {
      for (int j = 0; j < cpp_buffer_lists[i].size(); ++j) {
        args->output_lists[i][j] = new PJRT_Buffer{
            std::move(cpp_buffer_lists[i][j]), args->executable->client};
      }
    }
  } else {
    if (args->num_devices != 1) {
      return new PJRT_Error{xla::InvalidArgument(
          "num_devices and corresponding output list sizes must be 1 when "
          "calling PJRT_LoadedExecutable_Execute with non-null execute_device. "
          "Got "
          "num_devices=%i",
          args->num_devices)};
    }
    if (!cpp_send_callbacks->empty() || !cpp_recv_callbacks->empty()) {
      return new PJRT_Error{xla::Unimplemented(
          "PJRT_Executable_Execute doesn't support using send/recv callbacks "
          "with `execute_device`.")};
    }

    std::vector<std::unique_ptr<xla::PjRtBuffer>> cpp_buffer_list;
    std::optional<xla::PjRtFuture<xla::Status>> returned_future;
    bool fill_future = args->device_complete_events != nullptr;
    
    xla::CompileOptions compile_options = 
        args->executable->get()->GetCompileOptions().value_or(xla::CompileOptions());
    // PJRT_ASSIGN_OR_RETURN(xla::CompileOptions compile_options,
    //                       args->executable->get()->GetCompileOptions());
    if (compile_options.compile_portable_executable) {
      PJRT_ASSIGN_OR_RETURN(
          cpp_buffer_list,
          args->executable->get()->ExecutePortable(
              cpp_argument_lists[0], args->execute_device->device, options,
              returned_future, fill_future));
    } else {
      PJRT_ASSIGN_OR_RETURN(
          cpp_buffer_list,
          args->executable->get()->ExecuteSharded(
              cpp_argument_lists[0], args->execute_device->device, options,
              returned_future, fill_future));
    }
    for (int i = 0; i < cpp_buffer_list.size(); ++i) {
      args->output_lists[0][i] = new PJRT_Buffer{std::move(cpp_buffer_list[i]),
                                                 args->executable->client};
    }
    if (fill_future) {
      args->device_complete_events[0] =
          new PJRT_Event{std::move((*returned_future))};
    }
  }

  return nullptr;
}








constexpr static PJRT_Api CreatePjrtApi() {
    PJRT_Api api = pjrt::CreatePjrtApi(pjrt::plugin::PJRT_Client_Create,
                                       pjrt::plugin::PJRT_CpuDeviceTopology_Create);
    // api.PJRT_LoadedExecutable_Execute = pjrt::plugin::PJRT_LoadedExecutable_Execute;
    return api;
}

} // namespace pjrt::plugin

constexpr PJRT_Api pjrt_api = pjrt::plugin::CreatePjrtApi();
    // pjrt::CreatePjrtApi(pjrt::plugin::PJRT_Client_Create,
    //                     pjrt::plugin::PJRT_CpuDeviceTopology_Create);

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
