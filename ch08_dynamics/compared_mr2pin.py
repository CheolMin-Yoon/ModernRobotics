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
from params.ur5e import *  # noqa: F403

np.set_printoptions(precision=6, suppress=True)

UR5E_URDF = URDF_PATH  # from params.ur5e

model, data = load_urdf(UR5E_URDF)

# ── 테스트 설정 ──
q = np.array(thetalist, dtype=float)
dq = np.array([0.1, 0.2, -0.1, 0.05, 0.3, -0.2])
ddq = np.array([0.01, 0.02, -0.01, 0.005, 0.03, -0.02])
g_vec = np.array([0, 0, -9.81])

test_configs = {
    "home config": np.array(thetalist, dtype=float),
    "zero config": np.zeros(6),
    "random config": np.array([0.3, -1.2, 0.8, -0.5, 1.1, -0.7]),
}

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
            np.block([[I_joint, np.zeros((3, 3))],
                      [np.zeros((3, 3)), m_pin * np.eye(3)]]))

# ── 2. 역동역학 토크 (RNEA) ──
print("\n[2] 역동역학 토크 (RNEA)")
tau_pin = pin_rnea(model, data, q, dq, ddq)
tau_mr = RNEA(q, dq, ddq, g_vec, np.zeros(6), Mlist, Glist, Slist_space_vec)
print(f"  MR  tau: {tau_mr}")
print(f"  Pin tau: {tau_pin}")
compare("RNEA tau", tau_mr, tau_pin)

# ── 3. 질량 행렬 M(q) ──
print("\n[3] 질량 행렬 M(q)")
for name, q_cfg in test_configs.items():
    M_pin = pin_mass_matrix(model, data, q_cfg)
    M_mr = MassMatrix(q_cfg, Mlist, Glist, Slist_space_vec)
    print(f"\n  [{name}]")
    compare(f"{name} M(q)", M_mr, M_pin)

# ── 4. 중력 토크 ──
print("\n[4] 중력 토크 g(q)")
for name, q_cfg in test_configs.items():
    g_pin = pin_gravity(model, data, q_cfg)
    g_mr = GravityForces(q_cfg, g_vec, Mlist, Glist, Slist_space_vec)
    print(f"\n  [{name}]")
    compare(f"{name} g(q)", g_mr, g_pin)

# ── 5. 코리올리 + 중력 ──
print("\n[5] 비선형 효과 h(q,dq) = C*dq + g")
nle_pin = pin_nle(model, data, q, dq)
c_mr = VelQuadraticForces(q, dq, Mlist, Glist, Slist_space_vec)
g_mr = GravityForces(q, g_vec, Mlist, Glist, Slist_space_vec)
h_mr = c_mr + g_mr
print(f"  MR  h: {h_mr}")
print(f"  Pin h: {nle_pin}")
compare("h(q,dq)", h_mr, nle_pin)

# ── 6. 분해 검증: tau = M*ddq + h ──
print("\n[6] 분해 검증: tau = M*ddq + c + g")
M_mr = MassMatrix(q, Mlist, Glist, Slist_space_vec)
tau_decomp = M_mr @ ddq + h_mr
print(f"  M*ddq + h: {tau_decomp}")
print(f"  RNEA tau:  {tau_mr}")
compare("decomposition", tau_decomp, tau_mr, tol=1e-8)

print("\n" + "=" * 60)
print("  done")
print("=" * 60)