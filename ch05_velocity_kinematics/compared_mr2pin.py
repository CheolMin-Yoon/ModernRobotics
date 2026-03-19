# -*- coding: utf-8 -*-
"""
ch05 자코비안: MR 직접 구현 vs Pinocchio (URDF UR5) 비교 검증
conda env: mr
python ch05_velocity_kinematics/compared_mr2pin.py
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pinocchio as pin
from ch03_rigid_body_motion.modern_robotics_ch03 import (
    Adjoint, Vec2se3, TransInv
)
from ch05_velocity_kinematics.modern_robotics_ch05 import (
    BodyJacobian, SpaceJacobian
)

np.set_printoptions(precision=6, suppress=True)

# ── Pinocchio 모델 로드 ──
UR5_URDF = os.path.join(os.path.dirname(__file__), '..',
    'urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/'
    'universal_robots/ur_description/urdf/ur5.urdf')

model = pin.buildModelFromUrdf(UR5_URDF)
data = model.createData()
TOOL_FRAME = model.getFrameId('tool0')

# ── Zero config에서 M, 스크류 추출 (ch04 비교 코드와 동일) ──
q_zero = np.zeros(model.nq)
pin.forwardKinematics(model, data, q_zero)
pin.updateFramePlacements(model, data)
M = data.oMf[TOOL_FRAME].homogeneous.copy()

pin.computeJointJacobians(model, data, q_zero)
Slist_space_vec = []
for i in range(1, model.njoints):
    J = pin.getJointJacobian(model, data, i, pin.ReferenceFrame.WORLD)
    S_pin = J[:, i - 1]  # pin: [v, w]
    w = S_pin[3:].copy()
    v = S_pin[:3].copy()
    Slist_space_vec.append(np.concatenate([w, v]))  # MR: [w, v]

M_inv = TransInv(M)
Ad_Minv = Adjoint(M_inv)
Blist_body_vec = [Ad_Minv @ S for S in Slist_space_vec]


# ── Pinocchio 자코비안 (MR 컨벤션으로 변환) ──
def pin_jacobians(model, data, q):
    """Pinocchio 프레임 자코비안을 MR [w,v] 컨벤션으로 변환하여 반환"""
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    pin.computeJointJacobians(model, data, q)

    # 공간 자코비안: WORLD frame
    J_world = pin.computeFrameJacobian(
        model, data, q, TOOL_FRAME, pin.ReferenceFrame.WORLD)
    # pin [v,w] → MR [w,v]
    J_s_pin = np.vstack([J_world[3:, :], J_world[:3, :]])

    # 물체 자코비안: LOCAL (body) frame
    J_local = pin.computeFrameJacobian(
        model, data, q, TOOL_FRAME, pin.ReferenceFrame.LOCAL)
    J_b_pin = np.vstack([J_local[3:, :], J_local[:3, :]])

    return J_s_pin, J_b_pin


# ── 비교 함수 ──
def compare(name, mr_result, pin_result, tol=1e-4):
    diff = np.linalg.norm(np.asarray(mr_result) - np.asarray(pin_result))
    status = "PASS" if diff < tol else "FAIL"
    print(f"  [{name}] {status}  (diff={diff:.2e})")
    if diff >= tol:
        print(f"    MR :\n{np.asarray(mr_result)}")
        print(f"    Pin:\n{np.asarray(pin_result)}")


# ── 테스트 ──
print("=" * 60)
print("  ch05 Jacobian: MR vs Pinocchio (URDF UR5)")
print("=" * 60)

test_configs = {
    "zero config": np.zeros(6),
    "theta2=-90, theta5=90": np.array([0, -np.pi/2, 0, 0, np.pi/2, 0]),
    "all 45 deg": np.full(6, np.pi/4),
    "random config": np.array([0.3, -1.2, 0.8, -0.5, 1.1, -0.7]),
}

for name, q in test_configs.items():
    print(f"\n[{name}]  q = {np.round(q, 4)}")

    # MR 자코비안
    J_s_mr = SpaceJacobian(Slist_space_vec, q)
    J_b_mr = BodyJacobian(Blist_body_vec, q)

    # Pinocchio 자코비안
    J_s_pin, J_b_pin = pin_jacobians(model, data, q)

    compare("Space Jacobian (J_s)", J_s_mr, J_s_pin)
    compare("Body  Jacobian (J_b)", J_b_mr, J_b_pin)

print("\n" + "=" * 60)
print("  done")
print("=" * 60)
