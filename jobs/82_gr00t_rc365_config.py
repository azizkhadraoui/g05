"""GR00T N1.7 modality config for the RoboCasa365 LeRobot datasets (keys from their meta/modality.json).

The built-in ROBOCASA_PANDA_OMRON config targets NVIDIA's own RoboCasa export (res256_image_side_*, 12 state keys), so
RoboCasa365 is registered as NEW_EMBODIMENT with the keys and normalisation of the official RoboCasa365 GR00T N1.5 config
(robocasa-benchmark/Isaac-GR00T, PandaOmronDataConfig): 3 cameras, 5 state keys, 5 action keys, 16-step action chunks.
RoboCasa actions are already deltas/velocities (OSC end-effector deltas, base motion), so every action key is ABSOLUTE.
"""
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig

ACTION_KEYS = ["end_effector_position", "end_effector_rotation", "gripper_close", "base_motion", "control_mode"]

rc365_config = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]),
    "state": ModalityConfig(delta_indices=[0], modality_keys=[
        "end_effector_position_relative", "end_effector_rotation_relative", "gripper_qpos", "base_position", "base_rotation"]),
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=ACTION_KEYS,
        action_configs=[ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT)
                        for _ in ACTION_KEYS],
    ),
    "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.human.task_description"]),
}

register_modality_config(rc365_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
