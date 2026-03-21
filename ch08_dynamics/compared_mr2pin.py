# -*- coding: utf-8 -*-
"""
ch08 동역학: MR 직접 구현 vs Pinocchio (URDF UR5e) 비교 검증
conda env: mr
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from pin_utils.pin_utils import *
from ch08_dynamics.modern_robotics_ch08 import *
from ch05_velocity_kinematics.modern_robotics_ch05 import *
from ch04_forward_kinematics.modern_robotics_ch04 import thetalist

np.set_printoptions(precision=6, suppress=True)

UR5E_URDF = os.path.join(os.path.dirname(__file__), '..',
    'urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/'
    'universal_robots/ur_description/urdf/ur5e.urdf')

model, data = load_urdf(UR5E_URDF)

# ── 테스트 설정 ──
q = np.array(thetalist, dtype=float)
dthetalist = np.array([0.1, 0.2, -0.1, 0.05, 0.3, -0.2])
ddthetalist = np.array([0.01, 0.02, -0.01, 0.005, 0.03, -0.02])

print("=" * 60)
print("  ch08 동역학: MR vs Pinocchio (UR5e) via pin_utils")
print("=" * 60)

# ── 1. 공간 관성 행렬 비교 (각 링크) ──
print("\n[1] 공간 관성 행렬 G_b (각 링크)")
for i in range(6):
    G_mr = spatial_inertia(I_b[i], m[i])

    pin_inertia = model.inertias[i + 1]
    m_pin = pin_inertia.mass
    lever = pin_inertia.lever  # CoM offset
    I_joint = pin_inertia.inertia.copy()

    print(f"\n  --- link {i+1} ---")
    print(f"  MR  I_b diag: {I_b[i]}")
    print(f"  Pin I_joint:\n{I_joint}")
    print(f"  Pin lever (CoM): {lever}")
    print(f"  Pin mass: {m_pin}, MR mass: {m[i]}")

    compare(f"link {i+1} G_b (raw)", G_mr, 
            np.block([[I_joint, np.zeros((3,3))],
                      [np.zeros((3,3)), m_pin*np.eye(3)]]))

'''
# ── 2. 역동역학 토크 (RNEA) ──
print("\n[2] 역동역학 토크 (RNEA)")
tau_pin = pin_rnea(model, data, q, dthetalist, ddthetalist)
print(f"  Pinocchio tau: {tau_pin}")

# ── 3. 질량 행렬 M(q) ──
print("\n[3] 질량 행렬 M(q)")
M_pin = pin_mass_matrix(model, data, q)
print(f"  Pinocchio M(q):\n{M_pin}")

# ── 4. 코리올리 + 중력 ──
print("\n[4] 코리올리 + 중력 C(q,dq)*dq + g(q)")
nle_pin = pin_nle(model, data, q, dthetalist)
print(f"  Pinocchio nle: {nle_pin}")

# ── 5. 중력 토크 ──
print("\n[5] 중력 토크 g(q)")
g_pin = pin_gravity(model, data, q)
print(f"  Pinocchio g(q): {g_pin}")

print("\n" + "=" * 60)
print("  done")
print("=" * 60)
'''