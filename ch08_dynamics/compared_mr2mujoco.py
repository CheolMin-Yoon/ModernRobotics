# -*- coding: utf-8 -*-
"""
ch08 동역학: MR 직접 구현 vs MuJoCo (UR5e) 비교 검증
MuJoCo에서 UR5e를 로드하고, 관성/동역학 결과를 비교한다.

NOTE: MR(URDF)과 MuJoCo(menagerie MJCF)는 서로 다른 소스의 관성 파라미터를 사용한다.
  - 링크 로컬 프레임 축 방향이 다름 (관성 텐서 대각 원소 순서 차이)
  - wrist_3 질량 미세 차이 (URDF: 0.1879 vs MJCF: 0.1889)
  이는 좌표계 변환 버그가 아니라 데이터 소스 차이이므로,
  관성 파라미터 비교(섹션 1)는 참고용이고,
  동역학 결과(섹션 2~5)에는 tol=0.02를 적용한다.

conda env: mr
python ch08_dynamics/compared_mr2mujoco.py
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import mujoco
from ch08_dynamics.modern_robotics_ch08 import *
from params.ur5e import *  # noqa: F403

np.set_printoptions(precision=6, suppress=True)

# ── MuJoCo 모델 로드 ──
SCENE_XML = MUJOCO_SCENE  # from params.ur5e

mjmodel = mujoco.MjModel.from_xml_path(SCENE_XML)
mjdata = mujoco.MjData(mjmodel)

site_id = mujoco.mj_name2id(mjmodel, mujoco.mjtObj.mjOBJ_SITE, 'attachment_site')

joint_names = [mjmodel.joint(i).name for i in range(mjmodel.njnt)]
print(f"Joints: {joint_names}")


def set_joint_angles(data, q):
    for i in range(6):
        data.qpos[i] = q[i]


def set_joint_velocities(data, dq):
    for i in range(6):
        data.qvel[i] = dq[i]


def get_ee_pose(model, data):
    mujoco.mj_forward(model, data)
    pos = data.site_xpos[site_id].copy()
    rot = data.site_xmat[site_id].reshape(3, 3).copy()
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = pos
    return T


# ── 비교 함수 ──
def compare(name, mr_result, mj_result, tol=1e-4):
    diff = np.linalg.norm(np.asarray(mr_result) - np.asarray(mj_result))
    status = "✓ PASS" if diff < tol else "✗ FAIL"
    print(f"  [{name}] {status}  (diff={diff:.2e})")
    if diff >= tol:
        print(f"    MR    :\n{np.asarray(mr_result)}")
        print(f"    MuJoCo:\n{np.asarray(mj_result)}")


# URDF↔MJCF 관성 파라미터 차이에 의한 허용 오차
# (링크 프레임 축 방향 차이, wrist_3 질량 미세 차이 등)
DYNAMICS_TOL = 0.02
INERTIA_TOL = 0.2  # 관성 텐서는 축 순서가 달라 차이가 큼


# ── MuJoCo 역동역학 (RNEA 상당) ──
def mujoco_inverse_dynamics(model, data, q, dq, ddq):
    """MuJoCo mj_inverse로 역동역학 토크 계산"""
    set_joint_angles(data, q)
    set_joint_velocities(data, dq)
    for i in range(6):
        data.qacc[i] = ddq[i]
    mujoco.mj_inverse(model, data)
    return data.qfrc_inverse[:6].copy()


# ── MuJoCo 질량 행렬 ──
def mujoco_mass_matrix(model, data, q):
    """MuJoCo에서 질량 행렬 M(q) 추출"""
    set_joint_angles(data, q)
    mujoco.mj_forward(model, data)
    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, M, data.qM)
    return M[:6, :6].copy()


# ── MuJoCo 중력 토크 ──
def mujoco_gravity_torque(model, data, q):
    """중력 토크: dq=0, ddq=0에서의 역동역학"""
    return mujoco_inverse_dynamics(model, data, q,
                                   np.zeros(6), np.zeros(6))


# ── MuJoCo 코리올리 + 중력 ──
def mujoco_nle(model, data, q, dq):
    """비선형 효과 h(q,dq) = C(q,dq)*dq + g(q): ddq=0에서의 역동역학"""
    return mujoco_inverse_dynamics(model, data, q, dq, np.zeros(6))


# ── MuJoCo 관성 파라미터 추출 ──
def mujoco_body_inertia(model, body_name):
    """MuJoCo body의 질량, CoM, 관성 텐서 추출"""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    mass = model.body_mass[bid]
    com = model.body_ipos[bid].copy()       # CoM (부모 body 기준)
    # MuJoCo는 대각 관성 + 회전(quat)으로 저장
    diag_inertia = model.body_inertia[bid].copy()
    iquat = model.body_iquat[bid].copy()
    # 관성 텐서 복원: R @ diag(I) @ R^T
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, iquat)
    R = R.reshape(3, 3)
    I_full = R @ np.diag(diag_inertia) @ R.T
    return mass, com, I_full


# ══════════════════════════════════════════════════════════
#  테스트
# ══════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  ch08 동역학: MR vs MuJoCo (UR5e)")
print("=" * 60)

# ── 1. 관성 파라미터 비교 ──
# NOTE: URDF와 MJCF의 링크 로컬 프레임 축 방향이 달라 관성 텐서 대각 원소 순서가 다름.
#       이 비교는 참고용이며, 동역학 결과 비교(섹션 2~5)가 더 의미 있다.
print("\n[1] 관성 파라미터 (각 링크) — 참고: URDF↔MJCF 프레임 축 차이로 대각 순서 다름")
body_names = ["shoulder_link", "upper_arm_link", "forearm_link",
              "wrist_1_link", "wrist_2_link", "wrist_3_link"]

for i, bname in enumerate(body_names):
    m_mj, com_mj, I_mj = mujoco_body_inertia(mjmodel, bname)
    G_mr = spatial_inertia(I_b[i], m[i])
    G_mj = np.block([[I_mj,             np.zeros((3, 3))],
                     [np.zeros((3, 3)),  m_mj * np.eye(3)]])

    print(f"\n  --- {bname} ---")
    print(f"  MR  mass={m[i]:.4f}, I_diag={I_b[i]}")
    print(f"  MJ  mass={m_mj:.4f}, I_diag={np.diag(I_mj)}")
    print(f"  MJ  CoM={com_mj}")
    compare(f"{bname} G_b", G_mr, G_mj, tol=INERTIA_TOL)

# ── 2. 질량 행렬 M(q) ──
# NOTE: MuJoCo의 M(q)에는 joint armature (로터 관성)가 대각에 추가됨.
#       menagerie UR5e는 armature=0.1이 모든 조인트에 설정되어 있음.
#       URDF에는 armature가 없으므로, MR M(q) + diag(armature)로 비교한다.
print("\n[2] 질량 행렬 M(q) — armature 보정 포함")

# MuJoCo armature 추출
armature = np.array([mjmodel.dof_armature[i] for i in range(6)])
print(f"  MuJoCo armature: {armature}")

test_configs = {
    "home config": np.array(thetalist, dtype=float),
    "zero config": np.zeros(6),
    "random config": np.array([0.3, -1.2, 0.8, -0.5, 1.1, -0.7]),
}

for name, q in test_configs.items():
    M_mj = mujoco_mass_matrix(mjmodel, mjdata, q)
    M_mr = MassMatrix(q, Mlist, Glist, Slist_space_vec)
    M_mr_arm = M_mr + np.diag(armature)  # armature 보정
    print(f"\n  [{name}]")
    compare(f"{name} M(q)", M_mr_arm, M_mj, tol=DYNAMICS_TOL)

# ── 3. 역동역학 토크 (RNEA) ──
print("\n[3] 역동역학 토크 (RNEA)")
q_test = np.array(thetalist, dtype=float)
dq_test = np.array([0.1, 0.2, -0.1, 0.05, 0.3, -0.2])
ddq_test = np.array([0.01, 0.02, -0.01, 0.005, 0.03, -0.02])

g_vec = np.array([0, 0, -9.81])

tau_mj = mujoco_inverse_dynamics(mjmodel, mjdata, q_test, dq_test, ddq_test)
tau_mr = RNEA(q_test, dq_test, ddq_test, g_vec, np.zeros(6),
              Mlist, Glist, Slist_space_vec)
print(f"  MR    tau: {tau_mr}")
print(f"  MuJoCo tau: {tau_mj}")
compare("RNEA tau", tau_mr, tau_mj, tol=DYNAMICS_TOL)

# ── 4. 중력 토크 ──
print("\n[4] 중력 토크 g(q)")
for name, q in test_configs.items():
    g_mj = mujoco_gravity_torque(mjmodel, mjdata, q)
    g_mr = GravityForces(q, g_vec, Mlist, Glist, Slist_space_vec)
    print(f"\n  [{name}]")
    compare(f"{name} g(q)", g_mr, g_mj, tol=DYNAMICS_TOL)

# ── 5. 코리올리 + 중력 ──
print("\n[5] 비선형 효과 h(q,dq) = C*dq + g")
nle_mj = mujoco_nle(mjmodel, mjdata, q_test, dq_test)
c_mr = VelQuadraticForces(q_test, dq_test, Mlist, Glist, Slist_space_vec)
g_mr = GravityForces(q_test, g_vec, Mlist, Glist, Slist_space_vec)
h_mr = c_mr + g_mr
print(f"  MR    h: {h_mr}")
print(f"  MuJoCo h: {nle_mj}")
compare("h(q,dq)", h_mr, nle_mj, tol=DYNAMICS_TOL)

# ── 6. 분해 검증: tau = M*ddq + h ──
print("\n[6] 분해 검증: tau = M*ddq + c + g")
M_mr = MassMatrix(q_test, Mlist, Glist, Slist_space_vec)
tau_decomp = M_mr @ ddq_test + h_mr
print(f"  M*ddq + h: {tau_decomp}")
print(f"  RNEA tau:  {tau_mr}")
compare("decomposition", tau_decomp, tau_mr, tol=1e-8)

print("\n" + "=" * 60)
print("  done")
print("=" * 60)
