# -*- coding: utf-8 -*-
"""ch07 폐연쇄의 기구학 (Kinematics of Closed Chains)

3지 그리퍼(kinematics_pick_and_place) 기반 파지 해석:
- 파지 행렬 G (Grasp Matrix)
- 핸드 자코비안 J_h (Hand Jacobian)
- 능동/수동 관절 분리 → H 행렬로 자코비안 정리
- 폐연쇄 구속 조건 해석
- MuJoCo 시뮬레이션 기반 접촉 검증

참조: kinematics_pick_and_place/grasp_analysis.py (GraspAnalyzer)
"""

import numpy as np
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ch03_rigid_body_motion.modern_robotics_ch03 import *

PICK_PLACE_DIR = os.path.join(os.path.dirname(__file__), '..', 'kinematics_pick_and_place')


# ============================================================
# 7.1 파지 행렬 (Grasp Matrix)
# ============================================================

def grasp_matrix(p_contacts, p_obj, contact_type='point_friction'):
    """파지 행렬 G 계산

    접촉력 f_c → 물체 wrench F_o 매핑:
        F_o = G @ f_c

    Point contact with friction (3DOF/접촉점):
        G = [ I_3    I_3   ... ]   (6 x 3k)
            [ [r1×]  [r2×] ... ]
    여기서 r_i = p_contact_i - p_obj

    Args:
        p_contacts: 접촉점 위치 리스트 [(3,), ...], world frame
        p_obj:      물체 CoM 위치 (3,), world frame
        contact_type: 'point_friction' (3DOF) 또는 'rigid' (6DOF)

    Returns:
        G: 파지 행렬 (6 x m)
    """
    k = len(p_contacts)
    if contact_type == 'point_friction':
        dof_per_contact = 3
    elif contact_type == 'rigid':
        dof_per_contact = 6
    else:
        raise ValueError(f"Unknown contact type: {contact_type}")

    G = np.zeros((6, dof_per_contact * k))

    for i, p_c in enumerate(p_contacts):
        r = np.array(p_c) - np.array(p_obj)
        col = dof_per_contact * i

        if contact_type == 'point_friction':
            G[:3, col:col+3] = np.eye(3)
            G[3:, col:col+3] = Vec2so3(r)
        elif contact_type == 'rigid':
            G[:3, col:col+3] = np.eye(3)
            G[:3, col+3:col+6] = np.zeros((3, 3))
            G[3:, col:col+3] = Vec2so3(r)
            G[3:, col+3:col+6] = np.eye(3)

    return G


# ============================================================
# 7.2 핸드 자코비안 (Hand Jacobian)
# ============================================================

def hand_jacobian(finger_jacobians):
    """핸드 자코비안 J_h 계산

    각 손가락의 자코비안을 블록 대각으로 조합:
        J_h = blkdiag(J_f1, J_f2, ..., J_fk)

    Args:
        finger_jacobians: 각 손가락 자코비안 리스트 [J_f1, ..., J_fk]
                          J_fi: (m_i x n_i)

    Returns:
        J_h: 핸드 자코비안 (Σm_i x Σn_i)
    """
    from scipy.linalg import block_diag
    return block_diag(*finger_jacobians)


# ============================================================
# 7.3 능동/수동 관절 분리 및 H 행렬
# ============================================================

def partition_jacobian(J_full, active_idx, passive_idx):
    """자코비안을 능동/수동 관절로 분리

    전체 자코비안 J를 능동 관절(θ_a)과 수동 관절(θ_p)로 분리:
        V = J * dθ = J_a * dθ_a + J_p * dθ_p

    폐연쇄 구속 조건 하에서:
        dθ_p = -J_p^{-1} * J_a * dθ_a  =  H * dθ_a

    Args:
        J_full:      전체 자코비안 (m x n)
        active_idx:  능동 관절 인덱스 리스트
        passive_idx: 수동 관절 인덱스 리스트

    Returns:
        J_a: 능동 관절 자코비안 (m x n_a)
        J_p: 수동 관절 자코비안 (m x n_p)
        H:   수동-능동 매핑 행렬 (n_p x n_a), dθ_p = H * dθ_a
    """
    J_a = J_full[:, active_idx]
    J_p = J_full[:, passive_idx]
    H = -np.linalg.pinv(J_p) @ J_a
    return J_a, J_p, H


def closed_chain_jacobian(J_a, J_p, H):
    """폐연쇄 자코비안 계산

    능동 관절만으로 표현된 유효 자코비안:
        V = (J_a + J_p * H) * dθ_a = J_closed * dθ_a

    Args:
        J_a: 능동 관절 자코비안 (m x n_a)
        J_p: 수동 관절 자코비안 (m x n_p)
        H:   수동-능동 매핑 행렬 (n_p x n_a)

    Returns:
        J_closed: 폐연쇄 자코비안 (m x n_a)
    """
    return J_a + J_p @ H


# ============================================================
# 7.4 Force Closure 판별
# ============================================================

def check_force_closure(G, tol=1e-4):
    """Force closure 판별

    조건: rank(G) = 6 이고 최소 특이값 > tol

    Args:
        G:   파지 행렬 (6 x m)
        tol: 특이값 임계값

    Returns:
        is_closure: bool
        rank:       int
        min_sv:     float (최소 특이값)
        quality:    float (isotropy index = σ_min / σ_max)
    """
    if G is None:
        return False, 0, 0.0, 0.0

    U, S, Vt = np.linalg.svd(G)
    rank = np.sum(S > tol)
    min_sv = S[min(5, len(S)-1)] if len(S) > 0 else 0.0
    max_sv = S[0] if len(S) > 0 else 1.0
    quality = min_sv / max_sv if max_sv > tol else 0.0
    is_closure = (rank >= 6) and (min_sv > tol)

    return is_closure, rank, min_sv, quality


# ============================================================
# 7.5 Grübler 공식 기반 폐연쇄 자유도
# ============================================================

def grubler_dof(n_bodies, n_joints, joint_dofs, n_contact_constraints=0):
    """Grübler 공식: 폐연쇄 자유도 계산

    m = 6(N - 1 - J) + Σf_i - C

    Args:
        n_bodies:              링크 수 N (ground 포함)
        n_joints:              관절 수 J
        joint_dofs:            각 관절 자유도 리스트 [f_1, ..., f_J]
        n_contact_constraints: 접촉 구속 수 C

    Returns:
        dof: 폐연쇄 자유도
    """
    return 6 * (n_bodies - 1 - n_joints) + sum(joint_dofs) - n_contact_constraints


# ============================================================
# 7.6 3지 그리퍼 파지 종합 해석
# ============================================================

def three_finger_grasp_analysis(p_contacts, p_obj,
                                finger_jacobians,
                                active_idx, passive_idx):
    """3지 그리퍼 파지 종합 해석

    Args:
        p_contacts:        접촉점 위치 리스트 [(3,), ...]
        p_obj:             물체 CoM 위치 (3,)
        finger_jacobians:  각 손가락 자코비안 리스트
        active_idx:        능동 관절 인덱스 (전체 핸드 기준)
        passive_idx:       수동 관절 인덱스

    Returns:
        result: dict {
            'G':          파지 행렬 (6 x 3k),
            'J_h':        핸드 자코비안,
            'J_a':        능동 자코비안,
            'J_p':        수동 자코비안,
            'H':          수동-능동 매핑,
            'J_closed':   폐연쇄 자코비안,
            'force_closure': bool,
            'G_rank':     int,
            'quality':    float,
        }
    """
    G = grasp_matrix(p_contacts, p_obj)
    J_h = hand_jacobian(finger_jacobians)
    J_a, J_p, H = partition_jacobian(J_h, active_idx, passive_idx)
    J_closed = closed_chain_jacobian(J_a, J_p, H)
    is_closure, rank, min_sv, quality = check_force_closure(G)

    return {
        'G': G,
        'J_h': J_h,
        'J_a': J_a,
        'J_p': J_p,
        'H': H,
        'J_closed': J_closed,
        'force_closure': is_closure,
        'G_rank': rank,
        'min_singular_value': min_sv,
        'quality': quality,
    }


if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)
    print("=== ch07 폐연쇄 기구학: 3지 그리퍼 파지 해석 ===\n")

    # 간단한 테스트: 정육면체 물체를 3점에서 잡는 경우
    p_obj = np.array([0, 0, 0])
    p_contacts = [
        np.array([0.03, 0, 0]),    # center finger
        np.array([-0.015, 0.026, 0]),  # left finger
        np.array([-0.015, -0.026, 0]), # right finger
    ]

    G = grasp_matrix(p_contacts, p_obj)
    print(f"파지 행렬 G ({G.shape}):")
    print(G)
    print()

    is_closure, rank, min_sv, quality = check_force_closure(G)
    print(f"Force closure: {is_closure}")
    print(f"  rank(G) = {rank}, σ_min = {min_sv:.4f}, quality = {quality:.4f}")
    print()

    # 능동/수동 분리 테스트 (가상 자코비안)
    # 3 fingers × 4 joints = 12 joints, 3 contact DOF × 3 = 9 rows
    J_f1 = np.random.randn(3, 4)
    J_f2 = np.random.randn(3, 4)
    J_f3 = np.random.randn(3, 4)
    J_h = hand_jacobian([J_f1, J_f2, J_f3])
    print(f"핸드 자코비안 J_h ({J_h.shape})")

    # 능동: 각 손가락의 첫 2관절, 수동: 나머지 2관절
    active_idx = [0, 1, 4, 5, 8, 9]
    passive_idx = [2, 3, 6, 7, 10, 11]
    J_a, J_p, H = partition_jacobian(J_h, active_idx, passive_idx)
    J_closed = closed_chain_jacobian(J_a, J_p, H)
    print(f"J_a ({J_a.shape}), J_p ({J_p.shape}), H ({H.shape})")
    print(f"J_closed ({J_closed.shape})")
    print()

    # Grübler 자유도
    # 물체 자유 (접촉 전): 물체 6DOF + 손가락 12DOF = 18DOF
    # 접촉 구속: 3 point-contact × 3DOF = 9 스칼라 구속
    # → 18 - 9 = 9DOF (물체 + 그리퍼 전체 시스템)
    print("\n--- Grübler 자유도 ---")
    dof_free = grubler_dof(n_bodies=14, n_joints=12,
                           joint_dofs=[1]*12, n_contact_constraints=9)
    print(f"물체 자유 시스템 DOF: {dof_free}  (물체 6 + 손가락 12 - 접촉 9)")

    # 물체 고정 (ground에 weld): 접촉점이 고정된 점에 구속
    # N=13 (ground + 12 finger links), J=12 revolute, C=9 contact
    # → 6(13-1-12) + 12 - 9 = 0 + 3 = 3DOF (내부 자유도)
    dof_fixed = grubler_dof(n_bodies=13, n_joints=12,
                            joint_dofs=[1]*12, n_contact_constraints=9)
    print(f"물체 고정 시스템 DOF: {dof_fixed}  (손가락 내부 자유도만)")
