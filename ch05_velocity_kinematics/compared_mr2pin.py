# -*- coding: utf-8 -*-
"""
ch05 자코비안: MR 직접 구현 vs Pinocchio (URDF UR5) 비교 검증
conda env: mr
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from pin_utils.pin_utils import *
from ch03_rigid_body_motion.modern_robotics_ch03 import *
from ch05_velocity_kinematics.modern_robotics_ch05 import *

np.set_printoptions(precision=6, suppress=True)

UR5_URDF = os.path.join(os.path.dirname(__file__), '..',
    'urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/'
    'universal_robots/ur_description/urdf/ur5.urdf')

model, data = load_urdf(UR5_URDF)
TOOL_FRAME = model.getFrameId('tool0')

# ── Zero config에서 스크류 추출 ──
import pinocchio as pin

q_zero = np.zeros(model.nq)
M = pin_fk(model, data, q_zero, 'tool0')

pin.forwardKinematics(model, data, q_zero)
pin.computeJointJacobians(model, data, q_zero)
Slist_space_vec = []
for i in range(1, model.njoints):
    J = pin.getJointJacobian(model, data, i, pin.ReferenceFrame.WORLD)
    S_pin = J[:, i - 1]
    w = S_pin[3:].copy()
    v = S_pin[:3].copy()
    Slist_space_vec.append(np.concatenate([w, v]))

M_inv = TransInv(M)
Ad_Minv = Adjoint(M_inv)
Blist_body_vec = [Ad_Minv @ S for S in Slist_space_vec]


def pin_jacobians_mr(model, data, q):
    """Pinocchio 자코비안을 MR [w,v] 컨벤션으로 변환"""
    # 공간 자코비안 (WORLD)
    J_world = pin_jacobian(model, data, q, 'tool0', rf="world")
    J_s_pin = np.vstack([J_world[3:, :], J_world[:3, :]])

    # 물체 자코비안 (LOCAL)
    J_local = pin_jacobian(model, data, q, 'tool0', rf="local")
    J_b_pin = np.vstack([J_local[3:, :], J_local[:3, :]])

    return J_s_pin, J_b_pin


# ── 테스트 ──
print("=" * 60)
print("  ch05 Jacobian: MR vs Pinocchio (URDF UR5) via pin_utils")
print("=" * 60)

test_configs = {
    "zero config": np.zeros(6),
    "theta2=-90, theta5=90": np.array([0, -np.pi/2, 0, 0, np.pi/2, 0]),
    "all 45 deg": np.full(6, np.pi/4),
    "random config": np.array([0.3, -1.2, 0.8, -0.5, 1.1, -0.7]),
}

for name, q in test_configs.items():
    print(f"\n[{name}]  q = {np.round(q, 4)}")

    J_s_mr = SpaceJacobian(Slist_space_vec, q)
    J_b_mr = BodyJacobian(Blist_body_vec, q)

    J_s_pin, J_b_pin = pin_jacobians_mr(model, data, q)

    compare("Space Jacobian (J_s)", J_s_mr, J_s_pin)
    compare("Body  Jacobian (J_b)", J_b_mr, J_b_pin)

print("\n" + "=" * 60)
print("  done")
print("=" * 60)
