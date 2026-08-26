import os
import numpy as np
import mujoco
from types import SimpleNamespace
import torch as t
from motionbricks.motion_backbone.inference.motion_inference import motion_inference
from motionbricks.motion_backbone.demo.controllers import WASD_controller, random_controller
from motionbricks.exp_setup.experiment import test

class navigation_demo(object):
    def __init__(self, args):
        self.args = args
        self.full_agent = None
        self.controller = None
        self.mj_model = None
        self.mj_data = None
        self._parse_args()
        self._initialize_inference_modles()
        self._initialize_controller()
        self._initialize_mj_simulator()

    def _parse_args(self):
        self.args.return_model_configs = True
        self.args.return_dataloader = True

        # parse the default path if not given (very likely used by an external project)
        # Navigate from motionbricks/motion_backbone/demo/utils.py up to the project root
        project_base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        if not hasattr(self.args, 'humanoid_scene_xml'):
            self.args.humanoid_scene_xml = \
                os.path.abspath(os.path.join(project_base_path, "assets", "skeletons", "g1", "scene_29dof.xml"))

        if not hasattr(self.args, 'skeleton_xml'):
            self.args.skeleton_xml = \
                os.path.abspath(os.path.join(project_base_path, "assets", "skeletons", "g1", "g1.xml"))

        if not hasattr(self.args, 'result_dir'):
            self.args.result_dir = os.path.abspath(os.path.join(project_base_path, "out"))

        if not hasattr(self.args, 'data_root'):
            self.args.data_root = os.path.abspath(os.path.join(project_base_path, "datasets"))

        if not hasattr(self.args, 'clips_ckpt'):
            result_dir = getattr(self.args, 'result_dir', os.path.join(project_base_path, "out"))
            self.args.clips_ckpt = os.path.abspath(os.path.join(result_dir, "G1-clip.ckpt"))

        if not hasattr(self.args, 'explicit_dataset_folder'):
            self.args.explicit_dataset_folder = \
                os.path.abspath(os.path.join(project_base_path, "datasets", "motionbricks-G1"))

    def _initialize_inference_modles(self):
        reprocess_clips = getattr(self.args, 'reprocess_clips', False)  # useful for debugging & development
        if self.args.clips_ckpt is None or (not os.path.exists(self.args.clips_ckpt)) or reprocess_clips:
            models, confs, train_dataloader, val_dataloader = test(self.args)
            self.args.train_dataloader = train_dataloader
            self.args.val_dataloader = val_dataloader
        else:
            self.args.return_dataloader = False
            models, confs = test(self.args)
            self.args.train_dataloader = None
            self.args.val_dataloader = None

        for model_name in ['pose', 'root']:
            state_dict = t.load(confs[model_name].ckpt_path)['state_dict']
            models[model_name].load_state_dict(state_dict)
        self.inferencer = motion_inference(models, models['pose'].args)

        from motionbricks.motion_backbone.demo.full_agent import full_navigation_agent
        target_root_realignment = getattr(self.args, 'target_root_realignment', True)
        source_root_realignment = getattr(self.args, 'source_root_realignment', True)
        force_canonicalization = getattr(self.args, 'force_canonicalization', True)
        skip_ending_target_cond = getattr(self.args, 'skip_ending_target_cond', False)
        speed_scale = getattr(self.args, 'speed_scale', [0.8, 1.2]) if \
            getattr(self.args, 'random_speed_scale', False) else [1.0, 1.0]
        self.full_agent = full_navigation_agent(self.inferencer, self.args.train_dataloader, device='cuda',
                                                speed_scale=speed_scale,
                                                target_root_realignment=target_root_realignment,
                                                source_root_realignment=source_root_realignment,
                                                force_canonicalization=force_canonicalization,
                                                skeleton_xml=self.args.skeleton_xml,
                                                skip_ending_target_cond=skip_ending_target_cond,
                                                filter_qpos=getattr(self.args, 'pre_filter_qpos', True),
                                                clips=self.args.clips,
                                                ckpt_path=self.args.clips_ckpt,
                                                reprocess_clips=reprocess_clips,
                                                val_dataloader=self.args.val_dataloader).to('cuda')

    def _initialize_controller(self):
        lookat_movement_direction = getattr(self.args, 'lookat_movement_direction', False)
        min_tokens = self.inferencer._args['min_tokens']
        max_tokens = self.inferencer._args['max_tokens']

        if self.args.controller == "wasd":
            self.controller = WASD_controller(lookat_movement_direction=lookat_movement_direction,
                                              clips=self.args.clips, min_token=min_tokens, max_token=max_tokens)

        elif self.args.controller == "random":
            max_angle_change_between_controls = getattr(self.args, 'max_angle_change_between_controls', 0.5 * np.pi)
            self.controller = random_controller(disable_running=getattr(self.args, 'disable_running', True),
                                                lookat_movement_direction=lookat_movement_direction,
                                                new_control_dt=getattr(self.args, 'new_control_dt', 2.0),
                                                max_angle_change_between_controls=max_angle_change_between_controls,
                                                clips=self.args.clips, min_token=min_tokens, max_token=max_tokens)

        else:
            raise ValueError(f"Controller {self.args.controller} is not supported")

    def _initialize_mj_simulator(self):
        self.mj_model, self.mj_data = build_mj_simulator(self.args.humanoid_scene_xml, self.inferencer.motion_rep.fps)


def build_mj_simulator(humanoid_xml: str, fps: int = 30, build_dummy_mj_simulator: bool = False):
    if build_dummy_mj_simulator:
        # mj_model provide a value called qpos
        mj_model = SimpleNamespace(opt=SimpleNamespace(timestep=1 / fps))
        mj_data = SimpleNamespace(qpos=np.zeros(36))  # 36 is the number of qpos for humanoid of G1
    else:
        mj_model = mujoco.MjModel.from_xml_path(humanoid_xml)
        mj_data = mujoco.MjData(mj_model)
        # Disable advanced visual effects for better performance
        mj_model.vis.global_.offwidth = 1920
        mj_model.vis.global_.offheight = 1080
        mj_model.vis.quality.shadowsize = 0  # Disable shadows

        mj_model.vis.rgba.fog = [0, 0, 0, 0]  # Disable fog

        # Disable advanced lighting effects
        mj_model.vis.headlight.ambient = [0.8, 0.8, 0.8]  # Increase ambient light
        mj_model.vis.headlight.diffuse = [0.8, 0.8, 0.8]  # Increase diffuse light
        mj_model.vis.headlight.specular = [0.1, 0.1, 0.1]  # Reduce specular highlights

        mj_model.opt.timestep = 1 / fps
    return mj_model, mj_data


