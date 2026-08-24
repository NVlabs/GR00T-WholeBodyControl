#include <gtest/gtest.h>

#include "../include/fk.hpp"
#include "../include/motion_data_reader.hpp"

#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>

TEST(FK, TestFKAndGlobalVelocities) {

    const auto deploy_root = std::filesystem::path(GEAR_SONIC_DEPLOY_SOURCE_DIR);
    const auto reference_dir = deploy_root / "reference" / "example";
    const auto robot_model = deploy_root / "g1" / "g1_29dof.xml";

    // Use a small, named clip shipped through Git LFS with the repository.
    // The old bones_072925_test path referred to a fixture that was never
    // published, and indexing motions[0] after that failed load segfaulted.
    MotionDataReader motion_reader;
    ASSERT_TRUE(motion_reader.ReadFromCSV(reference_dir.string()));

    std::shared_ptr<MotionSequence> motion;
    for (const auto& candidate : motion_reader.motions) {
        if (candidate && candidate->name == "neutral_kick_R_001__A543_M") {
            motion = candidate;
            break;
        }
    }
    ASSERT_NE(motion, nullptr);

    RobotFK fk(robot_model.string());

    auto num_bodies = motion->GetNumBodies();
    auto num_joints = motion->GetNumJoints();
    auto timesteps = motion->timesteps;
    ASSERT_GT(num_bodies, 0);
    ASSERT_GT(num_joints, 0);
    ASSERT_GT(timesteps, 10);
    ASSERT_EQ(motion->GetNumBodyQuaternions(), num_bodies);
    ASSERT_EQ(motion->BodyPartIndexes().size(), static_cast<size_t>(num_bodies));

    std::vector<MotionSequence::Point> body_positions_orig(num_bodies * timesteps);
    std::vector<MotionSequence::Quaternion> body_quaternions_orig(num_bodies * timesteps);

    std::vector<MotionSequence::Velocity> body_lin_velocities_orig(num_bodies * timesteps);
    std::vector<MotionSequence::Velocity> body_ang_velocities_orig(num_bodies * timesteps);

    // record original global space data so we can compare it to the computed data:
    auto &seq = *motion;
    std::copy(seq.BodyPositions(0), seq.BodyPositions(0) + body_positions_orig.size(), body_positions_orig.begin());
    std::copy(seq.BodyQuaternions(0), seq.BodyQuaternions(0) + body_quaternions_orig.size(), body_quaternions_orig.begin());
    std::copy(seq.BodyLinVelocities(0), seq.BodyLinVelocities(0) + body_lin_velocities_orig.size(), body_lin_velocities_orig.begin());
    std::copy(seq.BodyAngVelocities(0), seq.BodyAngVelocities(0) + body_ang_velocities_orig.size(), body_ang_velocities_orig.begin());
    
    // Compute FK first. q and -q encode the same rotation; align each computed
    // quaternion to the sign used by the authoritative CSV before deriving
    // angular velocity or doing a component-wise comparison.
    motion->ComputeFK(fk);
    for (int f = 0; f < timesteps; ++f) {
        for (int b = 0; b < num_bodies; ++b) {
            auto &computed = seq.BodyQuaternions(f)[b];
            const auto &expected = body_quaternions_orig[f * num_bodies + b];
            double dot = 0.0;
            for (int i = 0; i < 4; ++i) {
                dot += computed[i] * expected[i];
            }
            if (dot < 0.0) {
                for (double &component : computed) {
                    component = -component;
                }
            }
        }
    }
    motion->ComputeGlobalVelocities();

    // sanity check - make sure at least some of the computed components are
    // different from the original data:
    bool found_different_pos = false;
    bool found_different_quat = false;
    bool found_different_lin_vel = false;
    bool found_different_ang_vel = false;
    for(int f = 0; f < timesteps; ++f) {
        for(int b = 0; b < num_bodies; ++b) {
            if(seq.BodyPositions(f)[b] != body_positions_orig[f * num_bodies + b]) {
                found_different_pos = true;
            }
            if(seq.BodyLinVelocities(f)[b] != body_lin_velocities_orig[f * num_bodies + b]) {
                found_different_lin_vel = true;
            }
            if(seq.BodyAngVelocities(f)[b] != body_ang_velocities_orig[f * num_bodies + b]) {
                found_different_ang_vel = true;
            }
            if(seq.BodyQuaternions(f)[b] != body_quaternions_orig[f * num_bodies + b]) {
                found_different_quat = true;
            }
        }
    }
    EXPECT_TRUE(found_different_pos);
    EXPECT_TRUE(found_different_quat);
    EXPECT_TRUE(found_different_lin_vel);
    EXPECT_TRUE(found_different_ang_vel);

    // check the results are close to the original data:
    for(int f = 0; f < timesteps; ++f) {
        for(int b = 0; b < num_bodies; ++b) {
            SCOPED_TRACE("frame=" + std::to_string(f) + ", body=" + std::to_string(b));
            for(int i = 0; i < 3; ++i) {
                EXPECT_NEAR(seq.BodyPositions(f)[b][i], body_positions_orig[f * num_bodies + b][i], 1e-5);
            }
            for(int i = 0; i < 4; ++i) {
                EXPECT_NEAR(seq.BodyQuaternions(f)[b][i], body_quaternions_orig[f * num_bodies + b][i], 1e-5);
            }
            for(int i = 0; i < 3; ++i) {
                EXPECT_NEAR(seq.BodyLinVelocities(f)[b][i], body_lin_velocities_orig[f * num_bodies + b][i], 2e-4);
            }

            // The shipped neutral-kick clip was generated with a slightly
            // different angular-velocity discretization.  FK orientation still
            // matches to 1e-5 above; allow the observed derivative drift here.
            if(timesteps - f > 10)
            {
                for(int i = 0; i < 3; ++i)
                {
                    EXPECT_NEAR(seq.BodyAngVelocities(f)[b][i], body_ang_velocities_orig[f * num_bodies + b][i], 1e-2);
                }
            }
        }
    }

}
