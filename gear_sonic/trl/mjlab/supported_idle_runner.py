"""Test-scaffold controller for atomic supported-idle PPO sessions.

Plain-mapping initialization and resume exist only for CPU unit tests.  Normal
construction has no route to an initialized controller.  Production execution
stays unavailable until a concrete adapter binds checkpoint paths and SHA-256
identities and captures the full MJLab environment and CUDA RNG state.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable, Mapping

from gear_sonic.trl.mjlab.supported_idle_checkpoint import (
    TensorSpec,
    build_checkpoint,
    capture_rng_state,
    restore_rng_state,
    save_checkpoint_exclusive,
    validate_lineage,
    validate_optimizer_semantics,
)

SESSION_UPDATE_CAP = 250
NUM_STEPS_PER_ENV = 24


def derive_session_seed(plan_payload_sha256: str, job_id: str, completed_updates: int) -> int:
    if (
        type(plan_payload_sha256) is not str
        or len(plan_payload_sha256) != 64
        or any(character not in "0123456789abcdef" for character in plan_payload_sha256)
    ):
        raise ValueError("plan_payload_sha256 must be lowercase SHA-256")
    if type(job_id) is not str or not job_id or type(completed_updates) is not int or completed_updates < 0:
        raise ValueError("invalid session seed inputs")
    digest = hashlib.sha256(f"{plan_payload_sha256}:{job_id}:{completed_updates}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


class SupportedIdleTrainingController:
    """Fail-closed CPU scaffold; production initialization is unavailable."""

    def __init__(
        self,
        *,
        alg: Any,
        env: Any,
        checkpoint_dir: str | Path,
        command_schema: Mapping[str, TensorSpec],
        lineage: Mapping[str, Any],
        optimizer_semantics: Mapping[str, Any],
        planned_updates: int,
        world_size: int = 1,
        checkpoint_builder: Callable[..., dict[str, Any]] | None = None,
        checkpoint_saver: Callable[..., Any] | None = None,
        rng_capture: Callable[[], dict[str, Any]] | None = None,
        rng_restore: Callable[[Mapping[str, Any]], None] | None = None,
        testing_only: bool = False,
    ) -> None:
        if type(planned_updates) is not int or planned_updates < 1:
            raise ValueError("planned_updates must be positive integer")
        if type(world_size) is not int or world_size != 1:
            raise ValueError("supported-idle controller requires one process")
        if not command_schema:
            raise ValueError("command_schema must not be empty")
        if type(testing_only) is not bool:
            raise TypeError("testing_only must be bool")
        self.alg = alg
        self.env = env
        self.checkpoint_dir = Path(checkpoint_dir)
        self.command_schema = command_schema
        self.lineage = validate_lineage(lineage)
        self.optimizer_semantics = validate_optimizer_semantics(optimizer_semantics)
        expected_initialization = {
            "dad_dance_seed": "seed_actor_critic_only",
            "qualified_phase_parent": "phase_parent_actor_critic_only",
        }[self.lineage["source_kind"]]
        if self.optimizer_semantics["initialization"] != expected_initialization:
            raise ValueError("lineage.source_kind contradicts optimizer_semantics")
        self.planned_updates = planned_updates
        custom_hooks = (checkpoint_builder, checkpoint_saver, rng_capture, rng_restore)
        if any(hook is not None for hook in custom_hooks) and testing_only is not True:
            raise ValueError("custom checkpoint/RNG hooks require testing_only=True")
        self._testing_only = testing_only
        self._checkpoint_builder = checkpoint_builder or build_checkpoint
        self._checkpoint_saver = checkpoint_saver or save_checkpoint_exclusive
        self._rng_capture = rng_capture or capture_rng_state
        # TODO: trusted MJLab adapter must capture/restore exact CUDA-device
        # RNG state. This CPU controller deliberately supplies only CPU default.
        self._rng_restore = rng_restore or restore_rng_state
        self.completed_updates = 0
        self.current_learning_iteration = 0
        self._poisoned = False
        self._initialized = False
        self._session_sealed = False
        self._checkpoint_saved = False

    def _require_healthy(self) -> None:
        if self._poisoned:
            raise RuntimeError(
                "controller poisoned by partial PPO update; discard it; production exact-resume adapter "
                "not implemented"
            )

    def _require_counter(self) -> int:
        if self.completed_updates != self.current_learning_iteration or self.completed_updates < 0:
            raise RuntimeError("controller update counters diverged")
        return self.completed_updates

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError(
                "controller is not initialized; production path/SHA and full environment-state adapter "
                "not implemented"
            )

    def _require_test_initialization_slot(self) -> None:
        if not self._testing_only:
            raise RuntimeError(
                "plain-mapping initialization/resume is testing-only; production path/SHA and full "
                "environment-state adapter not implemented"
            )
        if self._initialized or self._session_sealed or self._checkpoint_saved:
            raise RuntimeError("controller initialization already completed")

    def session_seed(self) -> int:
        return derive_session_seed(
            self.lineage["plan_payload_sha256"], self.lineage["job_id"], self.completed_updates
        )

    def _load_actor_critic(self, checkpoint: Mapping[str, Any]) -> None:
        self.alg.actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
        self.alg.critic.load_state_dict(checkpoint["critic_state_dict"], strict=True)

    def initialize_from_source(self, checkpoint: Mapping[str, Any], *, mode: str) -> None:
        """Test scaffold: load mapping actor+critic, then reset optimizer/counters."""

        self._require_healthy()
        self._require_test_initialization_slot()
        expected = {
            "seed_actor_critic_only": "dad_dance_seed",
            "phase_parent_actor_critic_only": "qualified_phase_parent",
        }
        if mode not in expected or self.lineage.get("source_kind") != expected[mode]:
            raise ValueError("initialization mode/lineage mismatch")
        reset = getattr(self.alg, "reset_optimizer", None)
        if not callable(reset):
            raise TypeError("algorithm must provide reset_optimizer()")
        if not isinstance(checkpoint, Mapping):
            raise ValueError("source checkpoint must be mapping")
        for key in ("actor_state_dict", "critic_state_dict"):
            if not isinstance(checkpoint.get(key), Mapping):
                raise ValueError(f"source checkpoint {key} must be mapping")
        try:
            self._load_actor_critic(checkpoint)
            reset()
            self.completed_updates = 0
            self.current_learning_iteration = 0
            self._initialized = True
        except BaseException:
            self._poisoned = True
            raise

    def resume_same_job(self, checkpoint: Mapping[str, Any]) -> None:
        """Test scaffold: restore a plain mapping without claiming exact resume."""

        self._require_healthy()
        self._require_test_initialization_slot()
        if not isinstance(checkpoint, Mapping):
            raise ValueError("resume checkpoint must be mapping")
        required_mappings = (
            "actor_state_dict",
            "critic_state_dict",
            "optimizer_state_dict",
            "rng_state",
            "trainer_state",
            "command_state",
            "lineage",
            "optimizer_semantics",
        )
        for key in required_mappings:
            if not isinstance(checkpoint.get(key), Mapping):
                raise ValueError(f"resume checkpoint {key} must be mapping")
        checkpoint_lineage = validate_lineage(checkpoint["lineage"])
        checkpoint_semantics = validate_optimizer_semantics(checkpoint["optimizer_semantics"])
        if checkpoint_lineage != self.lineage or checkpoint_semantics != self.optimizer_semantics:
            raise ValueError("same-job checkpoint lineage/optimizer semantics mismatch")
        trainer = checkpoint["trainer_state"]
        completed = trainer.get("completed_updates")
        current = trainer.get("current_learning_iteration")
        env_counter = trainer.get("env_common_step_counter")
        if (
            type(completed) is not int
            or completed < 1
            or type(current) is not int
            or current != completed
            or type(env_counter) is not int
            or env_counter < 0
        ):
            raise ValueError("same-job checkpoint trainer counters invalid")
        root_completed = checkpoint.get("completed_updates")
        root_iteration = checkpoint.get("iter")
        if (
            type(root_completed) is not int
            or root_completed != completed
            or type(root_iteration) is not int
            or root_iteration != completed - 1
        ):
            raise ValueError("same-job checkpoint root counters invalid")
        try:
            self._load_actor_critic(checkpoint)
            self.alg.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self._rng_restore(checkpoint["rng_state"])
            self.env.common_step_counter = env_counter
            self.env.load_command_state_dict(checkpoint["command_state"])
            self.completed_updates = completed
            self.current_learning_iteration = completed
            self._initialized = True
        except BaseException:
            self._poisoned = True
            raise

    def _checkpoint_path(self) -> Path:
        return self.checkpoint_dir / f"model_{self._require_counter()}.pt"

    def save_checkpoint(self) -> Any:
        self._require_healthy()
        self._require_initialized()
        if self._checkpoint_saved:
            raise RuntimeError("session checkpoint already saved")
        completed = self._require_counter()
        if completed < 1:
            raise RuntimeError("checkpoint capture requires a successful PPO update")
        try:
            checkpoint = self._checkpoint_builder(
                actor_state_dict=self.alg.actor.state_dict(),
                critic_state_dict=self.alg.critic.state_dict(),
                optimizer_state_dict=self.alg.optimizer.state_dict(),
                completed_updates=completed,
                env_common_step_counter=self.env.common_step_counter,
                lineage=self.lineage,
                optimizer_semantics=self.optimizer_semantics,
                rng_state=self._rng_capture(),
                command_state=self.env.command_state_dict(),
                command_schema=self.command_schema,
            )
            publication = self._checkpoint_saver(
                self._checkpoint_path(), checkpoint, command_schema=self.command_schema
            )
            self._checkpoint_saved = True
            return publication
        except BaseException:
            self._poisoned = True
            raise

    def learn(self, requested_updates: int) -> None:
        self._require_healthy()
        self._require_initialized()
        if self._session_sealed or self._checkpoint_saved:
            raise RuntimeError("training session already completed")
        if type(requested_updates) is not int or requested_updates < 1:
            raise ValueError("requested_updates must be positive integer")
        remaining = self.planned_updates - self._require_counter()
        maximum = min(SESSION_UPDATE_CAP, remaining)
        if requested_updates > maximum:
            raise ValueError("requested_updates exceeds session or planned cap")
        try:
            obs = self.env.get_observations()
            for _ in range(requested_updates):
                for _step in range(NUM_STEPS_PER_ENV):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions)
                    self.alg.process_env_step(obs, rewards, dones, extras)
                self.alg.compute_returns(obs)
                self.alg.update()
                self.completed_updates += 1
                self.current_learning_iteration = self.completed_updates
            self.save_checkpoint()
            self._session_sealed = True
        except BaseException:
            self._poisoned = True
            raise

    def export_policy_to_jit(self, *_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("supported-idle controller forbids JIT export")

    def export_policy_to_onnx(self, *_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("supported-idle controller forbids ONNX export")
