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
# Mlist & Glist: Pinocchio URDF 파싱 기반
# Pinocchio의 조인트 프레임을 MR 링크 프레임으로 사용하여
# MR RNEA == Pinocchio RNEA 일치를 보장한다.
# (기존 _frame_from_z 방식은 URDF→MR 프레임 변환 오류가 있었음)
# ============================================================
try:
    import pinocchio as pin

    _pin_model = pin.buildModelFromUrdf(
        os.path.join(os.path.dirname(__file__), '..',
            'urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/'
            'universal_robots/ur_description/urdf/ur5e.urdf'))
    _pin_data = _pin_model.createData()

    _q0 = np.zeros(_pin_model.nq)
    pin.forwardKinematics(_pin_model, _pin_data, _q0)
    pin.updateFramePlacements(_pin_model, _pin_data)

    _tool_fid = _pin_model.getFrameId('tool0')

    # ── T_{0,i}: Pinocchio joint placements at zero config ──
    _T0 = [np.eye(4)]  # base
    for _i in range(1, 7):
        _oMi = _pin_data.oMi[_i]
        _T = np.eye(4)
        _T[:3, :3] = _oMi.rotation
        _T[:3, 3] = _oMi.translation
        _T0.append(_T)
    # EE (tool0)
    _T_ee = np.eye(4)
    _T_ee[:3, :3] = _pin_data.oMf[_tool_fid].rotation
    _T_ee[:3, 3] = _pin_data.oMf[_tool_fid].translation
    _T0.append(_T_ee)

    # ── M_e 재정의 (Pinocchio tool0 기준) ──
    M_e = _T_ee.copy()

    # ── Slist: Pinocchio 공간꼴 자코비안에서 추출 ──
    _J_pin = pin.computeFrameJacobian(_pin_model, _pin_data, _q0, _tool_fid, pin.WORLD)
    Slist_space_vec = []
    for _i in range(6):
        _col = _J_pin[:, _i]
        # Pinocchio [v; w] → MR [w; v]
        Slist_space_vec.append(np.concatenate([_col[3:], _col[:3]]))
    Slist_space = [Vec2se3(S) for S in Slist_space_vec]

    # ── Blist: 물체꼴 스크류 축 ──
    _M_e_inv = TransInv(M_e)
    _Ad_Me_inv = Adjoint(_M_e_inv)
    Blist_body_vec = [_Ad_Me_inv @ S for S in Slist_space_vec]
    Blist_body = [Vec2se3(B) for B in Blist_body_vec]

    # ── Mlist: M_{i,i+1} = T_{0,i}^{-1} @ T_{0,i+1} ──
    Mlist = [TransInv(_T0[_i]) @ _T0[_i + 1] for _i in range(7)]

    # ── Glist: Pinocchio 관성 → MR 6×6 공간 관성 (조인트 프레임 원점 기준) ──
    Glist = []
    for _i in range(6):
        _inertia = _pin_model.inertias[_i + 1]
        _mi = _inertia.mass
        _lever = _inertia.lever      # CoM in joint frame
        _I_com = _inertia.inertia    # inertia at CoM, joint frame axes

        # 평행축 정리: I_origin = I_com + m*(|p|²I - p⊗p)
        _p = _lever
        _I_origin = _I_com + _mi * (np.dot(_p, _p) * np.eye(3) - np.outer(_p, _p))
        _p_skew = Vec2so3(_p)
        _G = np.block([[_I_origin,       _mi * _p_skew],
                       [_mi * _p_skew.T, _mi * np.eye(3)]])
        Glist.append(_G)

    # cleanup
    del _pin_model, _pin_data, _q0, _tool_fid, _T0, _T_ee, _J_pin
    del _M_e_inv, _Ad_Me_inv, _oMi, _T, _col, _inertia, _mi, _lever, _I_com, _p, _I_origin, _p_skew, _G, _i

except ImportError:
    # Pinocchio가 없으면 기존 수동 계산 방식 사용 (정확도 낮음)
    import warnings
    warnings.warn("Pinocchio not found. Using manual URDF params (may have frame conversion errors).")

    def _rpy2R(r, p, y):
        return MatrixExp3([0, 0, 1], y) @ MatrixExp3([0, 1, 0], p) @ MatrixExp3([1, 0, 0], r)

    def _frame_from_z(z_axis):
        z = np.array(z_axis, dtype=float)
        z = z / np.linalg.norm(z)
        if abs(z[0]) < 0.9:
            x = np.cross(z, [1, 0, 0])
        else:
            x = np.cross(z, [0, 1, 0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)
        return np.column_stack([x, y, z])

    _ws = [w1, w2, w3, w4, w5, w6]
    _qs = [q1, q2, q3, q4, q5, q6]
    _T0 = [np.eye(4)]
    for i in range(6):
        _T0.append(Rp2Trans(_frame_from_z(_ws[i]), _qs[i]))
    _T0.append(M_e)

    Mlist = [TransInv(_T0[i]) @ _T0[i + 1] for i in range(7)]

    Glist = []
    for i in range(6):
        mi = mass[i]
        G = np.block([[np.diag(inertia[i]), np.zeros((3, 3))],
                      [np.zeros((3, 3)),    mi * np.eye(3)]])
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
