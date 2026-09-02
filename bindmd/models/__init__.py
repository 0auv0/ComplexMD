from bindmd.models.bindmd import BindMD
from bindmd.models.flow import FlowBindMD
from bindmd.models.se3_torsion import (
    RigidFragmentSE3TorsionFlowBindMD,
    SE3TorsionFlowBindMD,
)
from bindmd.models.hierarchical import (
    HierarchicalPoseFlowBindMD,
    HierarchicalPoseRigidFragmentFlowBindMD,
    HierarchicalPoseSE3TorsionFlowBindMD,
)


def build_model(config: dict) -> BindMD:
    model_config = dict(config)
    generation_method = model_config.pop("generation_method", "diffusion")
    if generation_method in {
        "hierarchical_pose_rigid_fragment",
        "hierarchical_pose_rigid_fragment_flow",
    }:
        return HierarchicalPoseRigidFragmentFlowBindMD(**model_config)
    if generation_method in {
        "hierarchical_pose_se3_torsion",
        "hierarchical_pose_se3_torsion_flow",
    }:
        return HierarchicalPoseSE3TorsionFlowBindMD(**model_config)
    if generation_method in {"hierarchical_pose", "hierarchical_pose_flow"}:
        return HierarchicalPoseFlowBindMD(**model_config)
    if generation_method in {"se3_torsion", "se3_torsion_flow"}:
        return SE3TorsionFlowBindMD(**model_config)
    if generation_method in {"rigid_fragment", "rigid_fragment_flow"}:
        return RigidFragmentSE3TorsionFlowBindMD(**model_config)
    if generation_method in {"flow", "rectified_flow", "flow_matching"}:
        return FlowBindMD(**model_config)
    if generation_method != "diffusion":
        raise ValueError(f"unknown generation_method: {generation_method}")
    return BindMD(**model_config)


__all__ = [
    "BindMD", "FlowBindMD", "SE3TorsionFlowBindMD",
    "RigidFragmentSE3TorsionFlowBindMD", "HierarchicalPoseFlowBindMD",
    "HierarchicalPoseSE3TorsionFlowBindMD",
    "HierarchicalPoseRigidFragmentFlowBindMD",
    "build_model",
]
