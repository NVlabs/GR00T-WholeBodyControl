import time

import meshcat_shapes
import numpy as np
from pinocchio.visualize import MeshcatVisualizer

from decoupled_wbc.control.robot_model import RobotModel
from decoupled_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model


class RobotVisualizer:
    def __init__(self, robot: RobotModel):
        self.robot = robot
        self.viz = MeshcatVisualizer(
            self.robot.pinocchio_wrapper.model,
            self.robot.pinocchio_wrapper.collision_model,
            self.robot.pinocchio_wrapper.visual_model,
        )
        try:
            self.viz.initViewer(open=True)

        except ImportError as err:
            print("Error while initializing the viewer. It seems you should install Python meshcat")
            print(err)
            exit(0)

        self.viz.loadViewerModel()
        self.viz.display(self.robot.q_zero)

        # Visualize frames
        self.viz_frames = [self.robot.supplemental_info.root_frame_name]
        for side in ["left", "right"]:
            self.viz_frames.append(self.robot.supplemental_info.hand_frame_names[side])
        for frame in self.viz_frames:
            meshcat_shapes.frame(self.viz.viewer[frame], opacity=1.0)

    def visualize(self, robot_state: np.ndarray):
        # visualize robot state
        if robot_state is not None:
            self.robot.cache_forward_kinematics(robot_state, auto_clip=False)
            self.viz.display(robot_state)
            for frame_name in self.viz_frames:
                self.viz.viewer[frame_name].set_transform(self.robot.frame_placement(frame_name).np)


if __name__ == "__main__":
    # robot_model = instantiate_gr1_robot_model()
    robot_model = instantiate_g1_robot_model()
    visualizer = RobotVisualizer(robot_model)
    while True:
        visualizer.visualize(robot_model.q_zero)
        time.sleep(0.01)
