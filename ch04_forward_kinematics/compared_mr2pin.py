# -*- coding: utf-8 -*-
"""
ch04 FK 직접 구현 (PoE) vs Pinocchio (URDF UR5) 비교 검증
conda env: mr
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from pin_utils.pin_utils import *
from ch04_forward_kinematics.modern_robotics_ch04 import *
from ch03_rigid_body_motion.modern_robotics_ch03 import *

np.set_printoptions(precision=6, suppress=True)

UR5_URDF = os.path.join(os.path.dirname(__file__), '..',
    'urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/'
    'universal_robots/ur_description/urdf/ur5.urdf')

model, data = load_urdf(UR5_URDF)

# ── Zero config에서 M, 스크류 추출 ──
import pinocchio as pin

TOOL_FRAME = model.getFrameId('tool0')
q_zero = np.zeros(model.nq)

M = pin_fk(model, data, q_zero, 'tool0')

pin.forwardKinematics(model, data, q_zero)
pin.computeJointJacobians(model, data, q_zero)
Slist_space_vec = []
for i in range(1, model.njoints):
    J = pin.getJointJacobian(model, data, i, pin.ReferenceFrame.WORLD)
    S_pin = J[:, i-1]
    w = S_pin[3:].copy()
    v = S_pin[:3].copy()
    Slist_space_vec.append(np.concatenate([w, v]))

Slist_space = [Vec2se3(S) for S in Slist_space_vec]

M_inv = TransInv(M)
Ad_Minv = Adjoint(M_inv)
Blist_body_vec = [Ad_Minv @ S for S in Slist_space_vec]
Blist_body = [Vec2se3(B) for B in Blist_body_vec]


# ── 테스트 ──
print("=" * 55)
print("  ch04 FK: MR (PoE) vs Pinocchio (URDF UR5) via pin_utils")
print("=" * 55)

test_configs = {
    "zero config": np.zeros(6),
    "theta2=-90, theta5=90": np.array([0, -np.pi/2, 0, 0, np.pi/2, 0]),
    "all 45 deg": np.full(6, np.pi/4),
    "random config": np.array([0.3, -1.2, 0.8, -0.5, 1.1, -0.7]),
}

for name, q in test_configs.items():
    print(f"\n[{name}]  q = {np.round(q, 4)}")

    T_pin = pin_fk(model, data, q, 'tool0')
    T_space = fixed_frame_fk(Slist_space, q, M)
    T_body = body_frame_fk(Blist_body, q, M)

    compare("space FK vs Pin", T_space, T_pin)
    compare("body  FK vs Pin", T_body, T_pin)
    compare("space vs body", T_space, T_body, tol=1e-10)

print("\n" + "=" * 55)
print("  done")
print("=" * 55)
