#include <gtest/gtest.h>

#include "../include/control_policy.hpp"
#include "../include/encoder.hpp"
#include "../include/executorch_engine.hpp"

namespace {
// Tiny CPU programs written by unit_tests/fixtures/make_executorch_test_models.py.
const std::string kFixturesDir = EXECUTORCH_TEST_FIXTURES_DIR;
} // namespace

TEST(ExecuTorchEngineTest, PicksExecuTorchOnlyForPteFiles) {
  EXPECT_TRUE(IsExecuTorchModel("policy/model_decoder.pte"));
  EXPECT_FALSE(IsExecuTorchModel("policy/model_decoder.onnx"));
  EXPECT_FALSE(IsExecuTorchModel("policy/model.pte.onnx"));
  EXPECT_FALSE(IsExecuTorchModel("pte"));
}

#if HAS_EXECUTORCH
TEST(ExecuTorchEngineTest, RunsAProgramWithOneFloatInputAndOutput) {
  ExecuTorchEngine engine;
  ASSERT_TRUE(engine.Initialize(kFixturesDir + "/slice_add.pte"));
  EXPECT_EQ(engine.GetInputDimension(), 3u);
  EXPECT_EQ(engine.GetOutputDimension(), 2u);
  const float input[3] = {1.0f, 2.0f, 3.0f};
  float output[2] = {};
  ASSERT_TRUE(engine.Warmup(input));
  ASSERT_TRUE(engine.Run(input, output));
  EXPECT_FLOAT_EQ(output[0], 2.0f);
  EXPECT_FLOAT_EQ(output[1], 3.0f);
}

TEST(ExecuTorchEngineTest, RejectsProgramsWithOtherInputsOrOutputs) {
  for (const char* name : {"two_inputs", "int_input", "two_outputs", "int_output", "channels_last"}) {
    SCOPED_TRACE(name);
    ExecuTorchEngine engine;
    EXPECT_FALSE(engine.Initialize(kFixturesDir + "/" + name + ".pte"));
    EXPECT_EQ(engine.GetInputDimension(), 0u);
    EXPECT_EQ(engine.GetOutputDimension(), 0u);
    const float input[8] = {};
    float output[8] = {};
    EXPECT_FALSE(engine.Run(input, output));
  }
}

TEST(ExecuTorchEngineTest, FailedLoadDropsTheEarlierProgram) {
  ExecuTorchEngine engine;
  ASSERT_TRUE(engine.Initialize(kFixturesDir + "/slice_add.pte"));
  EXPECT_FALSE(engine.Initialize("does_not_exist.pte"));
  EXPECT_EQ(engine.GetInputDimension(), 0u);
  EXPECT_EQ(engine.GetOutputDimension(), 0u);
  const float input[3] = {1.0f, 2.0f, 3.0f};
  float output[2] = {};
  EXPECT_FALSE(engine.Run(input, output));
}

TEST(ExecuTorchEngineTest, EncoderEngineRunsAPteModel) {
  EncoderEngine encoder;
  ASSERT_TRUE(encoder.Initialize(kFixturesDir + "/slice_add.pte"));
  ASSERT_EQ(encoder.GetInputBuffer().size(), 3u);
  ASSERT_EQ(encoder.GetTokenBuffer().size(), 2u);
  encoder.GetInputBuffer()[0] = 1.0f;
  encoder.GetInputBuffer()[1] = 2.0f;
  encoder.GetInputBuffer()[2] = 3.0f;
  ASSERT_TRUE(encoder.CaptureGraph());
  ASSERT_TRUE(encoder.Encode());
  EXPECT_FLOAT_EQ(encoder.GetTokenBuffer()[0], 2.0f);
  EXPECT_FLOAT_EQ(encoder.GetTokenBuffer()[1], 3.0f);
}

TEST(ExecuTorchEngineTest, PolicyEngineRunsAPteModel) {
  PolicyEngine policy;
  ASSERT_TRUE(policy.Initialize(kFixturesDir + "/g1_policy.pte"));
  ASSERT_EQ(policy.GetInputBuffer().size(), 30u);
  ASSERT_EQ(policy.GetActionBuffer().size(), static_cast<size_t>(G1_NUM_MOTOR));
  for (size_t i = 0; i < policy.GetInputBuffer().size(); ++i) { policy.GetInputBuffer()[i] = static_cast<float>(i); }
  ASSERT_TRUE(policy.CaptureGraph());
  ASSERT_TRUE(policy.Infer());
  EXPECT_FLOAT_EQ(policy.GetActionBuffer()[0], 1.0f);
  EXPECT_FLOAT_EQ(policy.GetActionBuffer()[G1_NUM_MOTOR - 1], static_cast<float>(G1_NUM_MOTOR));
}

TEST(ExecuTorchEngineTest, EnginesDropTheEarlierModelOnAFailedLoad) {
  EncoderEngine encoder;
  ASSERT_TRUE(encoder.Initialize(kFixturesDir + "/slice_add.pte"));
  EXPECT_FALSE(encoder.Initialize("does_not_exist.pte"));
  EXPECT_FALSE(encoder.IsInitialized());
  PolicyEngine policy;
  ASSERT_TRUE(policy.Initialize(kFixturesDir + "/g1_policy.pte"));
  EXPECT_FALSE(policy.Initialize("does_not_exist.pte"));
  EXPECT_FALSE(policy.IsInitialized());
}

TEST(ExecuTorchEngineTest, PolicyEngineRejectsAPteWithTheWrongActionSize) {
  PolicyEngine policy;
  // 2 actions, while the G1 has G1_NUM_MOTOR motors.
  EXPECT_FALSE(policy.Initialize(kFixturesDir + "/slice_add.pte"));
}
#else
TEST(ExecuTorchEngineTest, FailsWithoutExecuTorchSupport) {
  ExecuTorchEngine engine;
  EXPECT_FALSE(engine.Initialize(kFixturesDir + "/slice_add.pte"));
  EXPECT_EQ(engine.GetInputDimension(), 0u);
  EXPECT_EQ(engine.GetOutputDimension(), 0u);
  EXPECT_FALSE(EncoderEngine().Initialize(kFixturesDir + "/slice_add.pte"));
  EXPECT_FALSE(PolicyEngine().Initialize(kFixturesDir + "/slice_add.pte"));
}
#endif
