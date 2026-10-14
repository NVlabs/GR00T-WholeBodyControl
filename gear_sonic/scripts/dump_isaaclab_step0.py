#!/usr/bin/env python3
"""Dump IsaacLab step-0 observations and policy intermediates for MuJoCo comparison.

This is a thin wrapper around ``gear_sonic.eval_agent_trl`` that monkey-patches
``UniversalTokenModule.forward`` to capture, on the first call only, all the
intermediate tensors needed to verify the MuJoCo deployment script:

    * Parsed tokenizer obs dict (every term)
    * Encoder input (concat of g1 inputs)
    * Encoder latent BEFORE FSQ quantization
    * Token AFTER FSQ quantization
    * Token-flattened (decoder input slice)
    * Proprioception (decoder input slice)
    * Decoder output (action_mean)
    * Robot ground-truth state at the SAME instant
      (joint_pos, joint_vel, root_pos, root_quat_wxyz, root_lin_vel_w,
       root_ang_vel_w) for every env

After the dump is written, ``sys.exit(0)`` is called so we don't waste time
running the full eval loop.

Usage:
    python gear_sonic/scripts/dump_isaaclab_step0.py \\
        +checkpoint=logs_rl/.../model_step_006000.pt \\
        +headless=True \\
        ++num_envs=1 \\
        ++eval_callbacks=im_eval \\
        ++run_eval_loop=true \\
        ++max_render_steps=1 \\
        ++dump_path=/tmp/x2_step0_isaaclab.pt
"""

import os
import sys
import pickle
import torch

# Ensure we can import the eval module (it lives under gear_sonic/)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


_DUMP_PATH = None
for _arg in sys.argv:
    if _arg.startswith("++dump_path=") or _arg.startswith("dump_path="):
        _DUMP_PATH = _arg.split("=", 1)[1]
        break
if _DUMP_PATH is None:
    _DUMP_PATH = "/tmp/x2_step0_isaaclab.pt"
sys.argv = [a for a in sys.argv if not (a.startswith("++dump_path=") or a.startswith("dump_path="))]

print(f"[dump_isaaclab_step0] Will dump to: {_DUMP_PATH}", flush=True)


def _install_forward_dump_patch():
    """Monkey-patch UniversalTokenModule.forward to capture step-0 tensors."""
    from gear_sonic.trl.modules import universal_token_modules as utm

    original_forward = utm.UniversalTokenModule.forward
    state = {"called": False}

    def patched_forward(self, input_data, compute_aux_loss=False, return_dict=False, **kwargs):
        if state["called"]:
            return original_forward(
                self, input_data, compute_aux_loss=compute_aux_loss,
                return_dict=return_dict, **kwargs
            )
        state["called"] = True
        print(f"[dump_isaaclab_step0] First UniversalTokenModule.forward call — capturing.",
              flush=True)

        # Reproduce the encode→quantize→decode logic so we can capture intermediates.
        batch_size, seq_len = input_data["actor_obs"].shape[:2]
        tokenizer_obs = self.parse_tokenizer_obs(input_data)
        proprioception_input = torch.cat(
            [input_data[key] for key in self.proprioception_features], dim=-1
        )

        # Run the g1 encoder on the LAST timestep only (mirrors evaluation use)
        # to get encoder input + latent + token.
        encoder_name = "g1"
        input_features = self.encoder_input_features[encoder_name]
        obs_list = [tokenizer_obs[key] for key in input_features]
        encoder_input_full = torch.cat(obs_list, dim=-1)  # (B, S, T_in, F_per_t)
        # Flatten batch+seq the same way the encoder does
        encoder_input_flat = encoder_input_full.view(-1, *encoder_input_full.shape[2:])
        # (B*S, num_input_temporal_dims, feature_per_t) -> (B*S, num_input_temporal_dims*feature_per_t)
        encoder_input_for_mlp = encoder_input_flat.view(encoder_input_flat.shape[0], -1)

        encoder = self.encoders[encoder_name]
        with torch.no_grad():
            latent = encoder(encoder_input_flat)  # (B*S, max_num_tokens, token_dim)

            quantized = None
            if self.quantizer is not None:
                quantized, _ = self.quantizer(latent)
            else:
                quantized = latent

            token_flattened = quantized.reshape(
                batch_size, seq_len, self.max_num_tokens * self.token_dim
            )

            # Decode through g1_dyn
            decoder_name = "g1_dyn"
            decoder = self.decoders[decoder_name]
            decoder_input = torch.cat(
                [token_flattened, proprioception_input], dim=-1
            )
            decoder_out = decoder(decoder_input)

        dump = {
            "actor_obs_shape": tuple(input_data["actor_obs"].shape),
            "tokenizer_shape": tuple(input_data["tokenizer"].shape),
            "actor_obs": input_data["actor_obs"].detach().cpu(),
            "tokenizer_flat": input_data["tokenizer"].detach().cpu(),
            "proprioception_input": proprioception_input.detach().cpu(),
            "tokenizer_obs": {
                k: v.detach().cpu() for k, v in tokenizer_obs.items()
            },
            "tokenizer_obs_dims": dict(self.tokenizer_obs_dims),
            "tokenizer_obs_names": list(self.tokenizer_obs_names),
            "encoder_input_features": dict(self.encoder_input_features),
            "encoder_input_full": encoder_input_full.detach().cpu(),
            "encoder_input_flat": encoder_input_flat.detach().cpu(),
            "encoder_input_for_mlp_view": encoder_input_for_mlp.detach().cpu(),
            "encoder_latent_g1": latent.detach().cpu(),
            "fsq_token_g1": quantized.detach().cpu(),
            "token_flattened": token_flattened.detach().cpu(),
            "decoder_input": decoder_input.detach().cpu(),
            "decoder_action_mean": decoder_out.detach().cpu(),
            "max_num_tokens": int(self.max_num_tokens),
            "token_dim": int(self.token_dim),
            "num_future_frames": int(self.num_future_frames),
            "proprioception_features": list(self.proprioception_features),
        }

        # Try to capture ground-truth robot state from the env via the global
        # reference saved by the eval entry point (see _capture_env_state).
        env_state = _ENV_STATE_HOLDER.get("state", None)
        if env_state is not None:
            dump["env_state"] = env_state
            print(f"[dump_isaaclab_step0] Captured env GT state with keys: "
                  f"{list(env_state.keys())}", flush=True)

        torch.save(dump, _DUMP_PATH)
        print(f"[dump_isaaclab_step0] Wrote dump to {_DUMP_PATH}", flush=True)
        print(f"  actor_obs shape: {dump['actor_obs_shape']}", flush=True)
        print(f"  tokenizer flat shape: {dump['tokenizer_shape']}", flush=True)
        print(f"  encoder_input_full shape: {tuple(encoder_input_full.shape)}", flush=True)
        print(f"  encoder_latent_g1 shape: {tuple(latent.shape)}", flush=True)
        print(f"  fsq_token_g1 shape: {tuple(quantized.shape)}", flush=True)
        print(f"  decoder_action_mean shape: {tuple(decoder_out.shape)}", flush=True)
        print(f"  decoder_action_mean[0]: {decoder_out[0].cpu().numpy()}", flush=True)

        # Bail out cleanly so we don't run the full eval
        os._exit(0)

    utm.UniversalTokenModule.forward = patched_forward
    print("[dump_isaaclab_step0] Installed forward dump patch on UniversalTokenModule.",
          flush=True)


_ENV_STATE_HOLDER = {"state": None}


def _install_env_state_patch():
    """Patch env.reset_all to capture GT robot state right after reset."""

    from gear_sonic import train_agent_trl

    original_create = train_agent_trl.create_manager_env

    def patched_create(*args, **kwargs):
        env = original_create(*args, **kwargs)
        original_reset_all = env.reset_all

        def patched_reset_all(*a, **kw):
            obs = original_reset_all(*a, **kw)
            try:
                # env.env is the IsaacLab ManagerBasedRLEnv
                inner_env = env.env if hasattr(env, "env") else env
                scene = inner_env.scene
                robot = scene["robot"]

                # robot.data exposes joint_pos, joint_vel, root_pos_w, root_quat_w (wxyz), etc.
                gt = {
                    "joint_pos": robot.data.joint_pos.detach().cpu(),
                    "joint_vel": robot.data.joint_vel.detach().cpu(),
                    "root_pos_w": robot.data.root_pos_w.detach().cpu(),
                    "root_quat_w_wxyz": robot.data.root_quat_w.detach().cpu(),
                    "root_lin_vel_w": robot.data.root_lin_vel_w.detach().cpu(),
                    "root_ang_vel_w": robot.data.root_ang_vel_w.detach().cpu(),
                    "root_lin_vel_b": robot.data.root_lin_vel_b.detach().cpu(),
                    "root_ang_vel_b": robot.data.root_ang_vel_b.detach().cpu(),
                    "joint_names": list(robot.data.joint_names),
                    "default_joint_pos": robot.data.default_joint_pos.detach().cpu(),
                }

                # Try to capture motion-command info too
                try:
                    cmd = inner_env.command_manager.get_term("motion")
                    gt["motion_ids"] = cmd.motion_ids.detach().cpu()
                    gt["motion_times"] = cmd.motion_times.detach().cpu()
                    if hasattr(cmd, "future_motion_ids"):
                        gt["future_motion_ids"] = cmd.future_motion_ids.detach().cpu()
                    if hasattr(cmd, "future_time_steps"):
                        gt["future_time_steps"] = cmd.future_time_steps.detach().cpu()
                    if hasattr(cmd, "num_future_frames"):
                        gt["num_future_frames"] = int(cmd.num_future_frames)
                except Exception as exc:  # noqa: BLE001
                    print(f"[dump_isaaclab_step0] Warning: motion command capture failed: {exc}",
                          flush=True)

                _ENV_STATE_HOLDER["state"] = gt
                print(f"[dump_isaaclab_step0] Captured GT env state. "
                      f"joint_pos[0,:6]={gt['joint_pos'][0,:6].numpy()}, "
                      f"root_pos_w[0]={gt['root_pos_w'][0].numpy()}, "
                      f"root_quat_w_wxyz[0]={gt['root_quat_w_wxyz'][0].numpy()}",
                      flush=True)
            except Exception as exc:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                print(f"[dump_isaaclab_step0] Warning: env state capture failed: {exc}",
                      flush=True)

            return obs

        env.reset_all = patched_reset_all
        return env

    train_agent_trl.create_manager_env = patched_create
    print("[dump_isaaclab_step0] Installed env state capture patch.", flush=True)


def main():
    # Install patches FIRST so any subsequent imports (and especially the env
    # creation) see the patched versions.
    _install_env_state_patch()
    _install_forward_dump_patch()

    # Hydra's @hydra.main resolves config_path relative to the script file that
    # owns the decorator. To keep that resolution working, we hand control to
    # the eval script via runpy as if it were the entry point — this is what
    # Hydra expects rather than importing main() from another module.
    import runpy
    eval_script_path = os.path.join(_REPO_ROOT, "gear_sonic", "eval_agent_trl.py")
    print(f"[dump_isaaclab_step0] Handing off to {eval_script_path}", flush=True)
    sys.argv[0] = eval_script_path
    runpy.run_path(eval_script_path, run_name="__main__")


if __name__ == "__main__":
    main()
