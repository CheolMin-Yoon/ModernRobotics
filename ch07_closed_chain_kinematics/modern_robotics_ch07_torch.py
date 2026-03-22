# -*- coding: utf-8 -*-
"""ch07 폐연쇄의 기구학 (Kinematics of Closed Chains) — PyTorch 버전

파지 해석, 핸드 자코비안, 폐연쇄 구속 조건 등을 PyTorch로 구현.
autograd를 통해 파지 품질 메트릭의 미분이 가능합니다.
"""

__all__ = [
    'grasp_matrix', 'hand_jacobian', 'partition_jacobian',
    'closed_chain_jacobian', 'check_force_closure',
    'grubler_dof', 'three_finger_grasp_analysis',
    'optimize_contact_positions',
]

import torch
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ch03_rigid_body_motion.modern_robotics_ch03_torch import Vec2so3


# ============================================================
# 7.1 파지 행렬 (Grasp Matrix)
# ============================================================

def grasp_matrix(p_contacts, p_obj, contact_type='point_friction'):
    """파지 행렬 G 계산: F_o = G @ f_c

    Args:
        p_contacts: 접촉점 위치 리스트 [tensor(3,), ...]
        p_obj:      물체 CoM 위치 tensor(3,)
        contact_type: 'point_friction' (3DOF) 또는 'rigid' (6DOF)

    Returns:
        G: 파지 행렬 (6 × m)
    """
    k = len(p_contacts)
    dtype = p_contacts[0].dtype
    device = p_contacts[0].device

    if contact_type == 'point_friction':
        dof = 3
    elif contact_type == 'rigid':
        dof = 6
    else:
        raise ValueError(f"Unknown contact type: {contact_type}")

    G = torch.zeros(6, dof * k, dtype=dtype, device=device)
    I3 = torch.eye(3, dtype=dtype, device=device)

    for i, p_c in enumerate(p_contacts):
        r = p_c - p_obj
        col = dof * i
        r_skew = Vec2so3(r)

        if contact_type == 'point_friction':
            G[:3, col:col+3] = I3
            G[3:, col:col+3] = r_skew
        elif contact_type == 'rigid':
            G[:3, col:col+3] = I3
            G[3:, col:col+3] = r_skew
            G[3:, col+3:col+6] = I3

    return G


# ============================================================
# 7.2 핸드 자코비안 (Hand Jacobian)
# ============================================================

def hand_jacobian(finger_jacobians):
    """핸드 자코비안: 블록 대각 조합"""
    total_rows = sum(J.shape[0] for J in finger_jacobians)
    total_cols = sum(J.shape[1] for J in finger_jacobians)
    dtype = finger_jacobians[0].dtype
    device = finger_jacobians[0].device

    J_h = torch.zeros(total_rows, total_cols, dtype=dtype, device=device)
    r, c = 0, 0
    for J in finger_jacobians:
        J_h[r:r+J.shape[0], c:c+J.shape[1]] = J
        r += J.shape[0]
        c += J.shape[1]
    return J_h


# ============================================================
# 7.3 능동/수동 관절 분리 및 H 행렬
# ============================================================

def partition_jacobian(J_full, active_idx, passive_idx):
    """자코비안을 능동/수동 관절로 분리

    Returns:
        J_a, J_p, H (dθ_p = H @ dθ_a)
    """
    J_a = J_full[:, active_idx]
    J_p = J_full[:, passive_idx]
    H = -torch.linalg.pinv(J_p) @ J_a
    return J_a, J_p, H


def closed_chain_jacobian(J_a, J_p, H):
    """폐연쇄 자코비안: V = (J_a + J_p @ H) @ dθ_a"""
    return J_a + J_p @ H


# ============================================================
# 7.4 Force Closure 판별
# ============================================================

def check_force_closure(G, tol=1e-4):
    """Force closure 판별

    Returns:
        is_closure, rank, min_sv, quality (isotropy index)
    """
    if G is None:
        return False, 0, torch.tensor(0.0), torch.tensor(0.0)

    S = torch.linalg.svdvals(G)
    rank = int(torch.sum(S > tol).item())
    min_sv = S[min(5, len(S)-1)] if len(S) > 0 else torch.tensor(0.0)
    max_sv = S[0] if len(S) > 0 else torch.tensor(1.0)
    quality = min_sv / max_sv if max_sv > tol else torch.tensor(0.0)
    is_closure = (rank >= 6) and (min_sv > tol)

    return is_closure, rank, min_sv, quality


# ============================================================
# 7.5 Grübler 공식
# ============================================================

def grubler_dof(n_bodies, n_joints, joint_dofs, n_contact_constraints=0):
    """Grübler 공식: m = 6(N-1-J) + Σf_i - C"""
    return 6 * (n_bodies - 1 - n_joints) + sum(joint_dofs) - n_contact_constraints


# ============================================================
# 7.6 autograd 기반 접촉점 최적화
# ============================================================

def optimize_contact_positions(p_contacts_init, p_obj, n_steps=200, lr=0.01):
    """autograd로 파지 품질(isotropy index)을 최대화하는 접촉점 위치 최적화

    σ_min / σ_max 를 최대화하도록 접촉점 위치를 경사 상승법으로 조정합니다.

    Args:
        p_contacts_init: 초기 접촉점 위치 리스트 [tensor(3,), ...]
        p_obj:           물체 CoM 위치 tensor(3,)
        n_steps:         최적화 스텝 수
        lr:              학습률

    Returns:
        optimized_contacts: 최적화된 접촉점 위치 리스트
        quality_history:    품질 변화 이력
    """
    # 접촉점을 최적화 변수로 설정
    contacts = [p.clone().detach().requires_grad_(True) for p in p_contacts_init]
    optimizer = torch.optim.Adam(contacts, lr=lr)
    quality_history = []

    for step in range(n_steps):
        optimizer.zero_grad()

        G = grasp_matrix(contacts, p_obj.detach())
        S = torch.linalg.svdvals(G)

        # 품질 = σ_min / σ_max (최대화 → loss = -quality)
        min_sv = S[-1]
        max_sv = S[0]
        quality = min_sv / (max_sv + 1e-8)
        loss = -quality

        loss.backward()
        optimizer.step()
        quality_history.append(quality.item())

    optimized = [p.detach() for p in contacts]
    return optimized, quality_history


# ============================================================
# 7.7 3지 그리퍼 파지 종합 해석
# ============================================================

def three_finger_grasp_analysis(p_contacts, p_obj,
                                finger_jacobians,
                                active_idx, passive_idx):
    """3지 그리퍼 파지 종합 해석"""
    G = grasp_matrix(p_contacts, p_obj)
    J_h = hand_jacobian(finger_jacobians)
    J_a, J_p, H = partition_jacobian(J_h, active_idx, passive_idx)
    J_closed = closed_chain_jacobian(J_a, J_p, H)
    is_closure, rank, min_sv, quality = check_force_closure(G)

    return {
        'G': G, 'J_h': J_h, 'J_a': J_a, 'J_p': J_p,
        'H': H, 'J_closed': J_closed,
        'force_closure': is_closure, 'G_rank': rank,
        'min_singular_value': min_sv, 'quality': quality,
    }


if __name__ == '__main__':
    torch.set_printoptions(precision=4, sci_mode=False)
    _dtype = torch.float64

    print("=== ch07 PyTorch 폐연쇄 기구학 테스트 ===\n")

    # 3점 접촉 파지
    p_obj = torch.tensor([0, 0, 0], dtype=_dtype)
    p_contacts = [
        torch.tensor([0.03, 0, 0], dtype=_dtype),
        torch.tensor([-0.015, 0.026, 0], dtype=_dtype),
        torch.tensor([-0.015, -0.026, 0], dtype=_dtype),
    ]

    G = grasp_matrix(p_contacts, p_obj)
    print(f"파지 행렬 G ({G.shape}):")
    print(G)

    is_closure, rank, min_sv, quality = check_force_closure(G)
    print(f"\nForce closure: {is_closure}")
    print(f"  rank(G) = {rank}, σ_min = {min_sv:.4f}, quality = {quality:.4f}")

    # 핸드 자코비안 테스트
    torch.manual_seed(42)
    J_f1 = torch.randn(3, 4, dtype=_dtype)
    J_f2 = torch.randn(3, 4, dtype=_dtype)
    J_f3 = torch.randn(3, 4, dtype=_dtype)
    J_h = hand_jacobian([J_f1, J_f2, J_f3])
    print(f"\n핸드 자코비안 J_h ({J_h.shape})")

    active_idx = [0, 1, 4, 5, 8, 9]
    passive_idx = [2, 3, 6, 7, 10, 11]
    J_a, J_p, H = partition_jacobian(J_h, active_idx, passive_idx)
    J_closed = closed_chain_jacobian(J_a, J_p, H)
    print(f"J_a ({J_a.shape}), J_p ({J_p.shape}), H ({H.shape})")
    print(f"J_closed ({J_closed.shape})")

    # Grübler 자유도
    print(f"\n--- Grübler 자유도 ---")
    dof = grubler_dof(14, 12, [1]*12, 9)
    print(f"물체 자유 시스템 DOF: {dof}")

    # autograd 접촉점 최적화
    print(f"\n--- autograd 접촉점 최적화 ---")
    p_init = [
        torch.tensor([0.02, 0.01, 0], dtype=_dtype),
        torch.tensor([-0.01, 0.02, 0], dtype=_dtype),
        torch.tensor([-0.01, -0.02, 0], dtype=_dtype),
    ]
    optimized, history = optimize_contact_positions(p_init, p_obj, n_steps=100, lr=0.001)
    print(f"초기 품질: {history[0]:.4f} → 최종 품질: {history[-1]:.4f}")
    for i, p in enumerate(optimized):
        print(f"  접촉점 {i}: {p}")
