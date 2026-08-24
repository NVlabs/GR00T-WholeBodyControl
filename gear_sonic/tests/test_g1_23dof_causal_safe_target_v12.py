from gear_sonic.scripts.train_g1_23dof_mjlab_causal_safe_target_v12 import (
    causal_safe_target_v12_training_contract,
)


def test_v12_is_bounded_stronger_v11_training() -> None:
    contract = causal_safe_target_v12_training_contract()
    assert contract["restart_from_model0"] is True
    assert contract["learning_rate"] == 5.0e-7
    assert contract["planned_accepted_updates"] == 100
    assert contract["allowed_checkpoint_updates"] == [0, 10, 25, 50, 100]
    assert contract["v11_transform_and_recovery_task_unchanged"] is True
    assert contract["deployment_ready"] is False
