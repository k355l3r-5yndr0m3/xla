/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/hlo/ir/dynamic_parameter_binding.h"

#include <memory>
#include <string>

#include "absl/algorithm/container.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/hlo_dce.h"
#include "xla/service/hlo_memory_scheduler.h"
#include "xla/service/hlo_ordering.h"
#include "xla/shape_util.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/types.h"
#include "tsl/lib/core/status_test_util.h"

namespace xla {
namespace {
using DynamicParameterBindingTest = HloTestBase;

TEST_F(DynamicParameterBindingTest, SimpleBinding) {
  // 'b' is a dynamic shape; 'a' represents the real size of b's first
  // dimension.
  const std::string module_str = R"(
HloModule TEST

ENTRY main {
  a = f32[] parameter(0)
  b = f32[10] parameter(1)
  ROOT root = (f32[], f32[10]) tuple(%a, %b)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));

  DynamicParameterBinding binding;

  TF_EXPECT_OK(
      binding.Bind(DynamicParameterBinding::DynamicParameter{0, {}},
                   DynamicParameterBinding::DynamicDimension{1, {}, 0}));

  auto test = [&](const DynamicParameterBinding& binding) {
    std::optional<DynamicParameterBinding::DynamicParameter> param =
        binding.GetBinding(
            DynamicParameterBinding::DynamicDimension{/*parameter_num=*/1,
                                                      /*parameter_index=*/{},
                                                      /*dimension=*/0});
    EXPECT_TRUE(param);
    EXPECT_EQ(param->parameter_num, 0);
    EXPECT_EQ(param->parameter_index, ShapeIndex({}));
    TF_EXPECT_OK(binding.Verify(*module));
  };
  test(binding);
}

TEST_F(DynamicParameterBindingTest, TupleBinding) {
  // 'gte2' is a dynamic shape; 'gte1' represents the real size of gte2's first
  // dimension.
  const std::string module_str = R"(
HloModule TEST

ENTRY main {
  param = (f32[], f32[10]) parameter(0)
  gte1 = f32[] get-tuple-element(%param), index=0
  gte2 = f32[10] get-tuple-element(%param), index=1
  ROOT root = (f32[], f32[10]) tuple(%gte1, %gte2)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));

  DynamicParameterBinding binding;

  TF_EXPECT_OK(
      binding.Bind(DynamicParameterBinding::DynamicParameter{0, {0}},
                   DynamicParameterBinding::DynamicDimension{0, {1}, 0}));

  auto test = [&](const DynamicParameterBinding& binding) {
    std::optional<DynamicParameterBinding::DynamicParameter> param =
        binding.GetBinding(
            DynamicParameterBinding::DynamicDimension{/*parameter_num=*/0,
                                                      /*parameter_index=*/{1},
                                                      /*dimension=*/0});

    EXPECT_TRUE(param);
    EXPECT_EQ(param->parameter_num, 0);
    EXPECT_EQ(param->parameter_index, ShapeIndex({0}));
    TF_EXPECT_OK(binding.Verify(*module));
  };
  test(binding);
}

TEST_F(DynamicParameterBindingTest, TupleBindingWithMultiDimension) {
  // 'gte2' is a dynamic shape; 'gte1' represents the real size of gte2's both
  // dimensions.
  const std::string module_str = R"(
HloModule TEST

ENTRY main {
  param = (f32[], f32[10, 10]) parameter(0)
  gte1 = f32[] get-tuple-element(%param), index=0
  gte2 = f32[10, 10] get-tuple-element(%param), index=1
  ROOT root = (f32[], f32[10, 10]) tuple(%gte1, %gte2)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));

  DynamicParameterBinding binding;

  TF_EXPECT_OK(
      binding.Bind(DynamicParameterBinding::DynamicParameter{0, {0}},
                   DynamicParameterBinding::DynamicDimension{0, {1}, 0}));

  TF_EXPECT_OK(
      binding.Bind(DynamicParameterBinding::DynamicParameter{0, {0}},
                   DynamicParameterBinding::DynamicDimension{0, {1}, 1}));

  auto test = [&](const DynamicParameterBinding& binding) {
    std::optional<DynamicParameterBinding::DynamicParameter> param =
        binding.GetBinding(
            DynamicParameterBinding::DynamicDimension{/*parameter_num=*/0,
                                                      /*parameter_index=*/{1},
                                                      /*dimension=*/0});

    EXPECT_TRUE(param);
    EXPECT_EQ(param->parameter_num, 0);
    EXPECT_EQ(param->parameter_index, ShapeIndex({0}));

    std::optional<DynamicParameterBinding::DynamicParameter> param2 =

        binding.GetBinding(
            DynamicParameterBinding::DynamicDimension{/*parameter_num=*/0,
                                                      /*parameter_index=*/{1},
                                                      /*dimension=*/0});
    EXPECT_TRUE(param2);
    EXPECT_EQ(param2->parameter_num, 0);
    EXPECT_EQ(param2->parameter_index, ShapeIndex({0}));
    TF_EXPECT_OK(binding.Verify(*module));
  };

  test(binding);
}

}  // namespace
}  // namespace xla
