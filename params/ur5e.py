# -*- coding: utf-8 -*-
"""UR5e 기구학 + 동역학 파라미터 (ros-industrial URDF 기준, 단위: m)

기구학: ch04_ur5e, ch08에서 공통 참조
동역학: ch08 전용 (질량, 관성, CoM)
"""

import numpy as np
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ch03_rigid_body_motion.modern_robotics_ch03 import *

# ============================================================
# 링크 파라미터 (단위: m)
# ============================================================
L1 = 0.425
L2 = 0.3922
W1 = 0.1333
W2 = 0.0997
H1 = 0.1625
H2 = 0.0996

# ============================================================
# 영 위치 EE 변환행렬 (M)
# ============================================================
R_e = np.array([[-1, 0,  0],
                [ 0, 0,  1],
                [ 0, 1,  0]])

p_e = np.array([[-(L1 + L2)],
                [-(W1 - H2 + W2)],
                [H1]])

M_e = np.block([[R_e,              p_e],
                [np.zeros((1, 3)), np.array([[1]])]])

# ============================================================
# 공간꼴 스크류 축 (S)
# ============================================================
w1 = np.array([0, 0, 1])
w2 = np.array([0, 1, 0])
w3 = np.array([0, 1, 0])
w4 = np.array([0, 1, 0])
w5 = np.array([0, 0, -1])
w6 = np.array([0, 1, 0])

q1 = np.array([0, 0, H1])
q2 = np.array([0, 0, H1])
q3 = np.array([-L1, 0, H1])
q4 = np.array([-(L1 + L2), 0, H1])
q5 = np.array([-(L1 + L2), -W1, H1])
q6 = np.array([-(L1 + L2), -(W1 - H2), H1])

v1 = -Vec2so3(w1) @ q1
v2 = -Vec2so3(w2) @ q2
v3 = -Vec2so3(w3) @ q3
v4 = -Vec2so3(w4) @ q4
v5 = -Vec2so3(w5) @ q5
v6 = -Vec2so3(w6) @ q6

Slist_space_vec = [np.concatenate([w, v]) for w, v in
                   zip([w1, w2, w3, w4, w5, w6],
                       [v1, v2, v3, v4, v5, v6])]

Slist_space = [Vec2se3(S) for S in Slist_space_vec]

# ============================================================
# 물체꼴 스크류 축 (B)
# ============================================================

M_e_inv = TransInv(M_e)
Ad_Me_inv = Adjoint(M_e_inv)

Blist_body_vec = [Ad_Me_inv @ S for S in Slist_space_vec]
Blist_body = [Vec2se3(B) for B in Blist_body_vec]

# ============================================================
# 동역학 RNEA, CRBA 관련 파라미터
# ============================================================

# 각 링크의 질량 (kg)
mass = [3.7, 8.393, 2.275, 1.219, 1.219, 0.1879]

# 각 링크의 CoM (링크 로컬 프레임 기준)
com = [
    np.array([0, 0, 0]),
    np.array([-0.2125, 0, 0.138]),
    np.array([-0.1961, 0, 0.007]),
    np.array([0, 0, 0]),
    np.array([0, 0, 0]),
    np.array([0, 0, -0.0229]),
]

# 각 링크의 관성 텐서 [Ixx, Iyy, Izz] (조인트 프레임 기준, rpy 반영 후)
inertia = [
    np.array([0.010267495893, 0.010267495893, 0.00666]),
    np.array([0.0151074, 0.1338857818, 0.1338857818]),
    np.array([0.004095, 0.0312093551, 0.0312093551]),
    np.array([0.0025598990, 0.0025598990, 0.0021942]),
    np.array([0.0025598990, 0.0025598990, 0.0021942]),
    np.array([9.890410e-05, 9.890410e-05, 1.321172e-04]),
]

# ============================================================
# Mlist & Glist: MR 교재 방식
# 각 링크 프레임: z축 = w_i (공간꼴 스크류 축 방향), 위치 = q_i
# → A_i = [0,0,1,0,0,0] 이 되어 Slist와 일관성 유지
# ============================================================
def _rpy2R(r, p, y):
    """URDF rpy → 회전행렬 (extrinsic XYZ = Rz@Ry@Rx)"""
    return MatrixExp3([0, 0, 1], y) @ MatrixExp3([0, 1, 0], p) @ MatrixExp3([1, 0, 0], r)

def _frame_from_z(z_axis):
    """z축 방향으로부터 직교 프레임 R 생성"""
    z = np.array(z_axis, dtype=float)
    z = z / np.linalg.norm(z)
    if abs(z[0]) < 0.9:
        x = np.cross(z, [1, 0, 0])
    else:
        x = np.cross(z, [0, 1, 0])
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])

# URDF 조인트 프레임 (rpy 누적)
_Tj = [np.eye(4)]
_Tj.append(_Tj[0] @ Rp2Trans(_rpy2R(0, 0, 0),                     [0,    0,   H1]))
_Tj.append(_Tj[1] @ Rp2Trans(_rpy2R(np.pi/2, 0, 0),               [0,    0,   0 ]))
_Tj.append(_Tj[2] @ Rp2Trans(_rpy2R(0, 0, 0),                     [-L1,  0,   0 ]))
_Tj.append(_Tj[3] @ Rp2Trans(_rpy2R(0, 0, 0),                     [-L2,  0,   W1]))
_Tj.append(_Tj[4] @ Rp2Trans(_rpy2R(np.pi/2, 0, 0),               [0,   -W2,  0 ]))
_Tj.append(_Tj[5] @ Rp2Trans(_rpy2R(np.pi/2, np.pi, np.pi),       [0,    W2,  0 ]))
_Tj.append(_Tj[6] @ Rp2Trans(_rpy2R(0, -np.pi/2, -np.pi/2),       [0,    0,   0 ]))

# MR 링크 프레임: z축 = w_i, 위치 = q_i
_ws = [w1, w2, w3, w4, w5, w6]
_qs = [q1, q2, q3, q4, q5, q6]
_T0 = [np.eye(4)]  # base
for i in range(6):
    _T0.append(Rp2Trans(_frame_from_z(_ws[i]), _qs[i]))
_T0.append(M_e)  # EE

# ── Mlist: M_{i,i+1} = T_{0,i}^{-1} @ T_{0,i+1} ──
M_01 = TransInv(_T0[0]) @ _T0[1]
M_12 = TransInv(_T0[1]) @ _T0[2]
M_23 = TransInv(_T0[2]) @ _T0[3]
M_34 = TransInv(_T0[3]) @ _T0[4]
M_45 = TransInv(_T0[4]) @ _T0[5]
M_56 = TransInv(_T0[5]) @ _T0[6]
M_6e = TransInv(_T0[6]) @ _T0[7]

Mlist = [M_01, M_12, M_23, M_34, M_45, M_56, M_6e]

# ── Glist: MR 링크 프레임 기준 6x6 공간 관성 (평행축 정리) ──
# URDF 프레임 → MR 프레임 회전 적용 후, CoM 오프셋으로 평행축 정리
Glist = []
for i in range(6):
    R_mr = _frame_from_z(_ws[i])       # MR 프레임 방향
    R_urdf = _Tj[i + 1][:3, :3]       # URDF 프레임 방향
    R_change = R_mr.T @ R_urdf         # URDF → MR 회전

    # 관성 텐서: MR 프레임 기준
    I_urdf = np.diag(inertia[i])
    I_mr = R_change @ I_urdf @ R_change.T

    # CoM: MR 프레임 기준
    com_mr = R_change @ com[i]

    # 평행축 정리
    mi = mass[i]
    p = com_mr
    p_skew = Vec2so3(p)
    I_joint = I_mr + mi * (np.dot(p, p) * np.eye(3) - np.outer(p, p))
    G = np.block([[I_joint,        mi * p_skew],
                  [mi * p_skew.T,  mi * np.eye(3)]])
    Glist.append(G)

# ============================================================
# 관절 제한 (URDF <limit> 태그 기준)
# ============================================================
# 위치 제한 [rad]
q_lower = np.array([-2*np.pi, -2*np.pi, -np.pi, -2*np.pi, -2*np.pi, -2*np.pi])
q_upper = np.array([ 2*np.pi,  2*np.pi,  np.pi,  2*np.pi,  2*np.pi,  2*np.pi])

# 최대 관절 속도 [rad/s]
dq_max = np.array([np.pi, np.pi, np.pi, np.pi, np.pi, np.pi])

# 최대 토크 [Nm]
tau_max = np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0])

# ============================================================
# 경로
# ============================================================
URDF_PATH = os.path.join(os.path.dirname(__file__), '..',
    'urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/'
    'universal_robots/ur_description/urdf/ur5e.urdf')

MUJOCO_SCENE = os.path.join(os.path.dirname(__file__), '..',
    'mujoco_menagerie/universal_robots_ur5e/scene.xml')

# ============================================================
# 호환 alias (ch08 등에서 사용하던 이름)
# ============================================================
m = mass          # 각 링크 질량 리스트
I_b = inertia     # 각 링크 관성 텐서 리스트

# se3 행렬 버전 alias (ch04_ur5e에서 사용)
Blist_e_body = Blist_body
Blist_e_body_vec = Blist_body_vec
Slist_e_space = Slist_space
Slist_e_space_vec = Slist_space_vec

# 기본 테스트 config
thetalist = [0, -np.pi/2, 0, 0, np.pi/2, 0]
