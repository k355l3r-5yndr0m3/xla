# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

load("@tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar")
load("//xla:xla.bzl", "xla_cc_binary")
load("@hedron_compile_commands//:refresh_compile_commands.bzl", "refresh_compile_commands")

# Configuration file for an XLA plugin.
#
# please don't check in changes to this file. to prevent changes appearing
# in git status, use:
#
# git update-index --assume-unchanged tensorflow/compiler/xla/pjrt/plugin/BUILD
#
# To add additional devices to the XLA subsystem, add targets to the
# dependency list in the 'plugin' target. For instance:
#
#   deps = ["//xla/pjrt/plugin/example:plugin_lib"],
#
# ** Please don't remove this file - it is supporting some 3rd party plugins **

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "plugin",
    # linkshared = True,
    # linkopts = ['-static'],
    # alwayslink = True,
    srcs = [
        "plugin.cc",
        "plugin.h",
    ],
    deps = [
        "//xla/pjrt:pjrt_api",

        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc_impl",
        "//xla:xla_data_proto_cc_impl",
        
        "//xla/client:client_library",

        "//xla/pjrt:pjrt_stream_executor_client",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "//xla/pjrt/c:pjrt_c_api_helpers",
        "//xla/pjrt/c:pjrt_c_api_wrapper_impl",

        "//xla/service:cpu_plugin",
        "//xla/service:gpu_plugin",
        "//xla/service:platform_util",
        "//xla/service:hlo_proto_cc_impl",
        "//xla/service:memory_space_assignment_proto_cc_impl",
        "//xla/service/gpu:backend_configs_cc_impl",

        "//xla/stream_executor:dnn_proto_cc_impl",

        "@tsl//tsl/platform:env_impl",
        "@tsl//tsl/platform:tensor_float_32_utils",
        "@tsl//tsl/profiler/utils:time_utils_impl",
        
        "@tsl//tsl/profiler/backends/cpu:annotation_stack_impl",
        "@tsl//tsl/profiler/backends/cpu:traceme_recorder_impl",
        
        # "@tsl//tsl/protobuf:autotuning_proto_cc_impl",
        "@tsl//tsl/protobuf:protos_all_cc_impl",
        "@tsl//tsl/protobuf:dnn_proto_cc_impl",
        "@tsl//tsl/framework:allocator",
        "@tsl//tsl/framework:allocator_registry_impl",
        "@tsl//tsl/util:determinism",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_binary(
    name = "plugin_test",
    srcs = [
        "plugin_test.c",
    ],
    deps = [
        ":plugin",
    ],
)

xla_cc_binary(
    name = "PJRTPlugin",
    linkshared = True,
    deps = [
        ":plugin"
#        "//xla/pjrt/c:pjrt_c_api_cpu"

#        "//xla/pjrt/c:pjrt_c_api_gpu",
#        "//xla/service:gpu_plugin",
#
#        "//xla/stream_executor/cuda:cuda_stream",
#        "//xla/stream_executor/cuda:all_runtime",
#        "//xla/stream_executor/cuda:stream_executor_cuda",
    ],
)

cc_library(
    name = "compile_options",
    hdrs = [
        "compile_options_utils.h"
    ],
    srcs = [
        "compile_options_utils.cc",
    ],
    deps = [
        "//xla/pjrt:pjrt_executable",
        "//xla/stream_executor:stream_executor",
    ],
)


refresh_compile_commands(
    name = "refresh_compile_commands",

    # Specify the targets of interest.
    # For example, specify a dict of targets and any flags required to build.
    targets = {
        "//xla/pjrt/plugin:plugin": "",
    },
    # No need to add flags already in .bazelrc. They're automatically picked up.
    # If you don't need flags, a list of targets is also okay, as is a single target string.
    # Wildcard patterns, like //... for everything, *are* allowed here, just like a build.
      # As are additional targets (+) and subtractions (-), like in bazel query https://docs.bazel.build/versions/main/query.html#expressions
    # And if you're working on a header-only library, specify a test or binary target that compiles it.
)


load("//xla/pjrt/plugin:cc_static_library.bzl", "cc_static_library")
cc_library(
    name = "compops_impl",
    srcs = [ 
        "compile_options.cc" 
    ],
    hdrs = [ 
        "compile_options.h"
    ],
    deps = [
        "//xla/pjrt:pjrt_executable",
        "//xla:xla_proto_cc_impl",
        "//xla:xla_data_proto_cc_impl",
        "//xla/service:hlo_proto_cc_impl",
        "//xla/service:memory_space_assignment_proto_cc_impl",
        "//xla/service/gpu:backend_configs_cc_impl",
        "//xla/service/gpu:hlo_op_profile_proto_cc_impl",
        "//xla/stream_executor:dnn_proto_cc_impl",
        "@tsl//tsl/platform:env_impl",
        "@tsl//tsl/platform:tensor_float_32_utils",
        "@tsl//tsl/profiler/utils:time_utils_impl",
        "@tsl//tsl/profiler/backends/cpu:annotation_stack_impl",
        "@tsl//tsl/profiler/backends/cpu:traceme_recorder_impl",
        "//xla:autotuning_proto_cc_impl",
        "@tsl//tsl/protobuf:protos_all_cc_impl",
        "@tsl//tsl/protobuf:dnn_proto_cc_impl",
        "@tsl//tsl/framework:allocator",
        "@tsl//tsl/framework:allocator_registry_impl",
        "@tsl//tsl/platform:status",
        "@tsl//tsl/util:determinism",
        "@com_google_protobuf//:protobuf",
    ],
)
cc_static_library(
    name = "libcompops",
    deps = [
        "//xla/pjrt/plugin:compops_impl",
    ],
)
