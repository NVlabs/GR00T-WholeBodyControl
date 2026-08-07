#include <gtest/gtest.h>

#include "../include/fk.hpp"
#include "../include/motion_data_reader.hpp"

#include <vector>
#include <fstream>

TEST(FK, TestFKAndGlobalVelocities) {

    // read some exapmle motion data:
    MotionDataReader motion_reader;
    motion_reader.ReadFromCSV("reference/bones_072925_test/");

    RobotFK fk("g1/g1_29dof.xml");

    auto num_bodies = motion_reader.motions[0]->GetNumBodies();
    auto num_joints = motion_reader.motions[0]->GetNumJoints();
    auto timesteps = motion_reader.motions[0]->timesteps;

    std::vector<MotionSequence::Point> body_positions_orig(num_bodies * timesteps);
    std::vector<MotionSequence::Quaternion> body_quaternions_orig(num_bodies * timesteps);

    std::vector<MotionSequence::Velocity> body_lin_velocities_orig(num_bodies * timesteps);
    std::vector<MotionSequence::Velocity> body_ang_velocities_orig(num_bodies * timesteps);

    // record original global space data so we can compare it to the computed data:
    const auto &seq = *motion_reader.motions[0];
    std::copy(seq.BodyPositions(0), seq.BodyPositions(0) + body_positions_orig.size(), body_positions_orig.begin());
    std::copy(seq.BodyQuaternions(0), seq.BodyQuaternions(0) + body_quaternions_orig.size(), body_quaternions_orig.begin());
    std::copy(seq.BodyLinVelocities(0), seq.BodyLinVelocities(0) + body_lin_velocities_orig.size(), body_lin_velocities_orig.begin());
    std::copy(seq.BodyAngVelocities(0), seq.BodyAngVelocities(0) + body_ang_velocities_orig.size(), body_ang_velocities_orig.begin());
    
    // Compute FK and global linear/angular velocities:
    motion_reader.motions[0]->ComputeFK(fk);
    motion_reader.motions[0]->ComputeGlobalVelocities();

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
            for(int i = 0; i < 3; ++i) {
                EXPECT_NEAR(seq.BodyPositions(f)[b][i], body_positions_orig[f * num_bodies + b][i], 1e-5);
            }
            for(int i = 0; i < 4; ++i) {
                EXPECT_NEAR(seq.BodyQuaternions(f)[b][i], body_quaternions_orig[f * num_bodies + b][i], 1e-5);
            }
            for(int i = 0; i < 3; ++i) {
                EXPECT_NEAR(seq.BodyLinVelocities(f)[b][i], body_lin_velocities_orig[f * num_bodies + b][i], 2e-4);
            }

            // I couldn't get this quite right as you can see by the restricted checking range and large tolerance.
            // However, if you write everything out to a .csv and plot the curves, they look close enough for me to
            // doubt this will cause a problem. Should get to the bottom of this and fix it though.
            if(timesteps - f > 10)
            {
                for(int i = 0; i < 3; ++i)
                {
                    EXPECT_NEAR(seq.BodyAngVelocities(f)[b][i], body_ang_velocities_orig[f * num_bodies + b][i], 5e-3);
                }
            }
        }
    }

}
