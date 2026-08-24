"""Low-VRAM RSL-RL configuration for exact true23 MJLab training."""

from __future__ import annotations

from dataclasses import dataclass

from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


@dataclass
class True23SonicActorCfg(RslRlModelCfg):
    """Qualified custom model configuration consumed by RSL-RL 5."""

    hidden_dims: tuple[int, ...] = ()
    activation: str = "silu"
    obs_normalization: bool = False
    distribution_cfg: dict | None = None
    class_name: str = (
        "gear_sonic.trl.mjlab.true23_actor:True23SonicActorModel"
    )
    warm_start_path: str = "sonic_release/g1_23dof_rev_1_0_init.pt"
    tokenizer_obs_group: str = "tokenizer"
    proprioception_obs_group: str = "policy"

    def __post_init__(self) -> None:
        if self.distribution_cfg is None:
            self.distribution_cfg = {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            }


def true23_mjlab_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Return a conservative RTX-3070-Laptop/8-GB starting configuration.

    Environment count remains an environment override.  The launch scripts use
    64 environments for smoke and 128 for the initial production attempt.
    """

    return RslRlOnPolicyRunnerCfg(
        actor=True23SonicActorCfg(),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=8,
            learning_rate=3.0e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        obs_groups={
            "actor": ("tokenizer", "policy"),
            "critic": ("critic",),
        },
        experiment_name="sonic_g1_23dof_mjlab",
        run_name="true23_h10",
        logger="tensorboard",
        upload_model=False,
        clip_actions=10.0,
        save_interval=50,
        num_steps_per_env=16,
        max_iterations=10_001,
    )
