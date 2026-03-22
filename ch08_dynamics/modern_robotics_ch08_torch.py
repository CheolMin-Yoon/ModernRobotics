# -*- coding: utf-8 -*-
"""ch08 동역학 (Dynamics) — PyTorch 버전

RNEA, 질량 행렬, 코리올리/원심력, 중력 벡터를 PyTorch로 구현.
autograd를 통해 동역학 파라미터에 대한 미분이 가능하므로,
시스템 식별(System Identification)이나 궤적 최적화에 활용할 수 있습니다.
"""

__all__ = [
    'spatial_inertia', 'lie_bracket', 'calculate_wrench',
    'RNEA', 'MassMatrix', 'MassMatrixCRBA',
    'VelQuadraticForces', 'GravityForces',
    'forward_dynamics', 'inverse_dynamics_autograd',
]

import torch
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ch03_rigid_body_motion.modern_robotics_ch03_torch import *


def spatial_inertia(I_b, m):
    """공간 관성 행렬 G_b (6×6)"""
    dtype = I_b.dtype
    device = I_b.device
    G = torch.zeros(6, 6, dtype=dtype, device=device)
    G[:3, :3] = torch.diag(I_b)
    G[3:, 3:] = m * torch.eye(3, dtype=dtype, device=device)
    return G


def lie_bracket(V_b):
    """리 브라켓 [ad_V] (6×6)"""
    w = V_b[:3]
    v = V_b[3:]
    w_mat = Vec2so3(w)
    v_mat = Vec2so3(v)
    zeros = torch.zeros(3, 3, dtype=V_b.dtype, device=V_b.device)
    top = torch.cat([w_mat, zeros], dim=1)
    bot = torch.cat([v_mat, w_mat], dim=1)
    return torch.cat([top, bot], dim=0)


def calculate_wrench(G_b, dV_b, V_b):
    """렌치 공식: F = G @ dV - [ad_V]^T @ G @ V"""
    ad_V = lie_bracket(V_b)
    return G_b @ dV_b - ad_V.T @ G_b @ V_b


def RNEA(thetalist, dthetalist, ddthetalist, g, F_tip,
         Mlist, Glist, Slist):
    """역 뉴턴-오일러 동역학 (MR Algorithm 8.2)

    모든 입력이 torch.Tensor이므로 autograd 미분 가능.

    Args:
        thetalist:   관절 각도 (n,)
        dthetalist:  관절 속도 (n,)
        ddthetalist: 관절 가속도 (n,)
        g:           중력 벡터 (3,)
        F_tip:       EE 팁 렌치 (6,)
        Mlist:       인접 링크 변환행렬 리스트 [M_01, ..., M_ne], len=n+1
        Glist:       공간 관성 행렬 리스트 [G_1, ..., G_n], len=n
        Slist:       공간꼴 스크류 축 리스트 [S_1, ..., S_n], len=n

    Returns:
        tau: 관절 토크 (n,)
    """
    n = len(thetalist)
    dtype = thetalist.dtype
    device = thetalist.device

    # A_i: 링크 i 프레임 기준 스크류 축
    Alist = []
    T_0i = torch.eye(4, dtype=dtype, device=device)
    for i in range(n):
        T_0i = T_0i @ Mlist[i]
        Ai = Adjoint(TransInv(T_0i)) @ Slist[i]
        Alist.append(Ai)

    # Forward pass: V_i, Vd_i
    V = [torch.zeros(6, dtype=dtype, device=device)]
    Vd = [torch.cat([torch.zeros(3, dtype=dtype, device=device), -g])]

    T_list = []
    for i in range(n):
        Ai = Alist[i]
        Ti = MatrixExp6(Vec2se3(-Ai * thetalist[i])) @ TransInv(Mlist[i])
        T_list.append(Ti)

        AdTi = Adjoint(Ti)
        Vi = AdTi @ V[i] + Ai * dthetalist[i]
        Vdi = AdTi @ Vd[i] \
              + lie_bracket(Vi) @ Ai * dthetalist[i] \
              + Ai * ddthetalist[i]

        V.append(Vi)
        Vd.append(Vdi)

    # Backward pass: F_i, tau_i
    F = F_tip.clone()
    tau = torch.zeros(n, dtype=dtype, device=device)

    for i in range(n - 1, -1, -1):
        Gi = Glist[i]
        Ai = Alist[i]
        Vi = V[i + 1]
        Vdi = Vd[i + 1]

        if i == n - 1:
            T_next = TransInv(Mlist[n])
        else:
            T_next = T_list[i + 1]

        Fi = Gi @ Vdi - lie_bracket(Vi).T @ Gi @ Vi \
             + Adjoint(T_next).T @ F

        tau[i] = Fi @ Ai
        F = Fi

    return tau


def MassMatrix(thetalist, Mlist, Glist, Slist):
    """질량 행렬 M(θ) — RNEA n회 호출"""
    n = len(thetalist)
    dtype = thetalist.dtype
    device = thetalist.device
    M_mat = torch.zeros(n, n, dtype=dtype, device=device)

    dth_zero = torch.zeros(n, dtype=dtype, device=device)
    g_zero = torch.zeros(3, dtype=dtype, device=device)
    F_zero = torch.zeros(6, dtype=dtype, device=device)

    for i in range(n):
        ddth = torch.zeros(n, dtype=dtype, device=device)
        ddth[i] = 1.0
        M_mat[:, i] = RNEA(thetalist, dth_zero, ddth,
                            g_zero, F_zero, Mlist, Glist, Slist)
    return M_mat


def MassMatrixCRBA(thetalist, Mlist, Glist, Slist):
    """CRBA (Composite Rigid Body Algorithm) — O(n²)"""
    n = len(thetalist)
    dtype = thetalist.dtype
    device = thetalist.device

    # A_i, T_{i,i-1} 계산
    Alist = []
    T_0i = torch.eye(4, dtype=dtype, device=device)
    for i in range(n):
        T_0i = T_0i @ Mlist[i]
        Ai = Adjoint(TransInv(T_0i)) @ Slist[i]
        Alist.append(Ai)

    T_list = []
    for i in range(n):
        Ai = Alist[i]
        Ti = MatrixExp6(Vec2se3(-Ai * thetalist[i])) @ TransInv(Mlist[i])
        T_list.append(Ti)

    # Composite Inertia
    IC = [G.clone() for G in Glist]
    for i in range(n - 2, -1, -1):
        Ad = Adjoint(TransInv(T_list[i + 1]))
        IC[i] = IC[i] + Ad.T @ IC[i + 1] @ Ad

    # M 조립
    M_mat = torch.zeros(n, n, dtype=dtype, device=device)
    for i in range(n):
        M_mat[i, i] = Alist[i] @ IC[i] @ Alist[i]
        F = IC[i] @ Alist[i]
        for j in range(i + 1, n):
            F = Adjoint(TransInv(T_list[j])).T @ F
            M_mat[i, j] = Alist[j] @ F
            M_mat[j, i] = M_mat[i, j]

    return M_mat


def VelQuadraticForces(thetalist, dthetalist, Mlist, Glist, Slist):
    """코리올리/원심력 벡터 c(θ, dθ)"""
    n = len(thetalist)
    dtype = thetalist.dtype
    device = thetalist.device
    return RNEA(thetalist, dthetalist,
                torch.zeros(n, dtype=dtype, device=device),
                torch.zeros(3, dtype=dtype, device=device),
                torch.zeros(6, dtype=dtype, device=device),
                Mlist, Glist, Slist)


def GravityForces(thetalist, g, Mlist, Glist, Slist):
    """중력 벡터 g(θ)"""
    n = len(thetalist)
    dtype = thetalist.dtype
    device = thetalist.device
    return RNEA(thetalist,
                torch.zeros(n, dtype=dtype, device=device),
                torch.zeros(n, dtype=dtype, device=device),
                g,
                torch.zeros(6, dtype=dtype, device=device),
                Mlist, Glist, Slist)


def forward_dynamics(thetalist, dthetalist, tau, g, F_tip,
                     Mlist, Glist, Slist):
    """정동역학: ddθ = M^{-1} @ (τ - c - g - J^T F_tip)

    autograd를 통해 τ에 대한 ddθ의 미분도 가능합니다.
    """
    M_mat = MassMatrixCRBA(thetalist, Mlist, Glist, Slist)
    c = VelQuadraticForces(thetalist, dthetalist, Mlist, Glist, Slist)
    grav = GravityForces(thetalist, g, Mlist, Glist, Slist)

    # τ_total = τ - c - g  (F_tip은 RNEA 내부에서 처리)
    rhs = tau - c - grav
    ddthetalist = torch.linalg.solve(M_mat, rhs)
    return ddthetalist


def inverse_dynamics_autograd(thetalist, dthetalist, ddthetalist, g,
                              Mlist, Glist, Slist):
    """autograd 기반 역동역학 검증

    M(θ) @ ddθ + c(θ,dθ) + g(θ) 를 직접 계산하여 RNEA 결과와 비교.
    M(θ)를 autograd로 미분 가능한 형태로 구성합니다.
    """
    M_mat = MassMatrixCRBA(thetalist, Mlist, Glist, Slist)
    c = VelQuadraticForces(thetalist, dthetalist, Mlist, Glist, Slist)
    grav = GravityForces(thetalist, g, Mlist, Glist, Slist)
    return M_mat @ ddthetalist + c + grav


if __name__ == '__main__':
    import numpy as np

    torch.set_printoptions(precision=4, sci_mode=False)
    _dtype = torch.float64

    print("=== ch08 PyTorch 동역학 테스트 ===\n")

    # UR5e 파라미터 로드 (numpy → torch 변환)
    from params.ur5e import (
        Mlist as Mlist_np, Glist as Glist_np,
        Slist_space_vec as Slist_np,
        Blist_body_vec as Blist_np,
    )

    Mlist_t = [torch.tensor(m, dtype=_dtype) for m in Mlist_np]
    Glist_t = [torch.tensor(g, dtype=_dtype) for g in Glist_np]
    Slist_t = [torch.tensor(s, dtype=_dtype) for s in Slist_np]

    theta = torch.tensor([0, -torch.pi/2, 0, 0, torch.pi/2, 0], dtype=_dtype)
    dtheta = torch.tensor([0.1, 0.2, -0.1, 0.05, 0.3, -0.2], dtype=_dtype)
    ddtheta = torch.tensor([0.01, 0.02, -0.01, 0.005, 0.03, -0.02], dtype=_dtype)
    g = torch.tensor([0, 0, -9.81], dtype=_dtype)
    F_tip = torch.zeros(6, dtype=_dtype)

    # RNEA
    tau = RNEA(theta, dtheta, ddtheta, g, F_tip, Mlist_t, Glist_t, Slist_t)
    print("RNEA 토크:")
    print(tau)

    # 질량 행렬
    M_mat = MassMatrix(theta, Mlist_t, Glist_t, Slist_t)
    M_crba = MassMatrixCRBA(theta, Mlist_t, Glist_t, Slist_t)
    print(f"\nMassMatrix == CRBA: {torch.allclose(M_mat, M_crba, atol=1e-6)}")
    print("M(θ) 대각:")
    print(torch.diag(M_crba))

    # c, g 벡터
    c = VelQuadraticForces(theta, dtheta, Mlist_t, Glist_t, Slist_t)
    grav = GravityForces(theta, g, Mlist_t, Glist_t, Slist_t)
    print(f"\n코리올리/원심력: {c}")
    print(f"중력 벡터: {grav}")

    # 역동역학 검증: M@ddθ + c + g == RNEA(τ)
    tau_check = inverse_dynamics_autograd(theta, dtheta, ddtheta, g,
                                          Mlist_t, Glist_t, Slist_t)
    print(f"\nM@ddθ + c + g == RNEA: {torch.allclose(tau, tau_check, atol=1e-6)}")

    # autograd: τ에 대한 θ의 기울기
    theta_g = theta.clone().requires_grad_(True)
    tau_g = RNEA(theta_g, dtheta, ddtheta, g, F_tip, Mlist_t, Glist_t, Slist_t)
    tau_g.sum().backward()
    print(f"\n∂(Στ)/∂θ (autograd):")
    print(theta_g.grad)

    # 정동역학
    tau_input = tau.detach().clone()
    ddtheta_fd = forward_dynamics(theta, dtheta, tau_input, g, F_tip,
                                   Mlist_t, Glist_t, Slist_t)
    print(f"\n정동역학 ddθ:")
    print(ddtheta_fd)
    print(f"원래 ddθ와 일치: {torch.allclose(ddtheta_fd, ddtheta, atol=1e-4)}")
