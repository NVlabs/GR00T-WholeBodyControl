/**
 * @file fk.hpp
 * @brief Forward kinematics (FK) for the G1 robot, parsed from a MuJoCo XML model.
 *
 * RobotFK loads a MuJoCo MJCF XML file (e.g. `g1/g1_29dof.xml`), extracts the
 * kinematic tree (joint axes, translations, rest rotations), and provides a
 * `DoFK()` method that computes world-frame positions and orientations for
 * every body given a root pose and joint angles.
 *
 * The FK result is used by MotionSequence::ComputeFK() to populate body-part
 * positions and quaternions from joint-angle data.
 */

#pragma once

#include <string>
#include <vector>
#include <array>

struct XMLNode;

/**
 * @class RobotFK
 * @brief Forward kinematics solver for the G1 robot, initialised from a MuJoCo XML.
 */
class RobotFK
{
    public:

        RobotFK(const std::string &xmlfile);

        void DoFK(
            std::array<double, 3> *positions_world,
            std::array<double, 4> *rotations_world,
            const std::array<double, 3> &root_translation,
            const std::array<double, 4> &root_rotation,
            const double *joint_angles
        ) const;

        int NumJoints() const { return node_children_.size(); }
    
    private:

        void FKChildren(
            std::array<double, 3> *positions_world,
            std::array<double, 4> *rotations_world,
            int parent_idx,
            const double *joint_angles
        ) const;

        void AddNode(XMLNode *node);

        std::vector< std::array<double, 3> > axes_;
        std::vector< std::array<double, 3> > translations_;
        std::vector< std::array<double, 4> > rest_rotations_;
        std::vector< std::vector<int> > node_children_;
        std::vector< std::string > node_names_;
};
