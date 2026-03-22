# -*- coding: utf-8 -*-
"""ch05 속도 기구학과 정역학 (Velocity Kinematics) — PyTorch 버전

자코비안 계산이 autograd로 자동 미분 가능합니다.
또한 FK의 θ에 대한 자동 미분으로 자코비안을 구하는 autograd_jacobian도 제공합니다.
"""

__all__ = [
    'BodyJacobian', 'SpaceJacobian', 'autograd_body_jacobian',
]

import torch
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ch03_rigid_body_motion.modern_robotics_ch03_torch import *
from ch04_forward_kinematics.modern_robotics_ch04_torch import *


def BodyJacobian(Blist_vec, thetalist):
    """5.1 물체 자코비안 J_b (6×n)

    J_b[:,i] = Ad(e^{-[Bn]θn} ... e^{-[Bi+1]θi+1}) @ B_i
    """
    n = len(thetalist)
    dtype = Blist_vec[0].dtype
    device = Blist_vec[0].device
    J_b = torch.zeros(6, n, dtype=dtype, device=device)

    J_b[:, n - 1] = Blist_vec[n - 1]

    T = torch.eye(4, dtype=dtype, device=device)
    for i in range(n - 2, -1, -1):
        T = T @ MatrixExp6(Vec2se3(-Blist_vec[i + 1] * thetalist[i + 1]))
        J_b[:, i] = Adjoint(T) @ Blist_vec[i]

    return J_b


def SpaceJacobian(Slist_vec, thetalist):
    """5.2 공간 자코비안 J_s (6×n)

    J_s[:,i] = Ad(e^{[S1]θ1} ... e^{[Si-1]θi-1}) @ S_i
    """
    n = len(thetalist)
    dtype = Slist_vec[0].dtype
    device = Slist_vec[0].device
    J_s = torch.zeros(6, n, dtype=dtype, device=device)

    J_s[:, 0] = Slist_vec[0]

    T = torch.eye(4, dtype=dtype, device=device)
    for i in range(1, n):
        T = T @ MatrixExp6(Vec2se3(Slist_vec[i - 1] * thetalist[i - 1]))
        J_s[:, i] = Adjoint(T) @ Slist_vec[i]

    return J_s


def autograd_body_jacobian(Blist_body_se3, thetalist, M_mat):
    """autograd 기반 물체 자코비안 계산

    FK의 θ에 대한 자동 미분으로 자코비안을 구합니다.
    해석적 자코비안과 비교 검증에 유용합니다.

    Returns:
        J_pos: 위치 자코비안 (3×n), ∂p/∂θ
    """
    theta = thetalist.detach().clone().requires_grad_(True)
    T = body_frame_fk(Blist_body_se3, theta, M_mat)
    p = T[:3, 3]

    J_pos = torch.zeros(3, len(theta), dtype=theta.dtype, device=theta.device)
    for i in range(3):
        if theta.grad is not None:
            theta.grad.zero_()
        p[i].backward(retain_graph=True)
        J_pos[i] = theta.grad.clone()

    return J_pos


def autograd_full_jacobian(Blist_body_se3, thetalist, M_mat):
    """autograd 기반 전체 자코비안 (4×4 → 16 출력에 대한 미분)

    FK 출력 T의 모든 원소에 대한 θ 미분을 구합니다.
    Returns:
        J_full: (16×n) 자코비안
    """
    theta = thetalist.detach().clone().requires_grad_(True)
    T = body_frame_fk(Blist_body_se3, theta, M_mat)
    T_flat = T.reshape(-1)

    n = len(theta)
    J_full = torch.zeros(16, n, dtype=theta.dtype, device=theta.device)
    for i in range(16):
        if theta.grad is not None:
            theta.grad.zero_()
        T_flat[i].backward(retain_graph=True)
        J_full[i] = theta.grad.clone()

    return J_full


if __name__ == '__main__':
    torch.set_printoptions(precision=3, sci_mode=False)
    _dtype = torch.float64

    print("=== ch05 PyTorch 자코비안 테스트 ===\n")

    theta = torch.tensor(thetalist, dtype=_dtype)

    J_b = BodyJacobian(Blist_body_vec, theta)
    J_s = SpaceJacobian(Slist_space_vec, theta)

    print("물체 자코비안 J_b:")
    print(J_b)
    print("\n공간 자코비안 J_s:")
    print(J_s)

    # 검증: J_s = [Ad_Tsb] @ J_b
    T_sb = body_frame_fk(Blist_body, theta)
    Ad_Tsb = Adjoint(T_sb)
    J_s_from_Jb = Ad_Tsb @ J_b
    print(f"\n[Ad_Tsb] @ J_b == J_s: {torch.allclose(J_s, J_s_from_Jb, atol=1e-6)}")

    # autograd 기반 위치 자코비안
    J_pos = autograd_body_jacobian(Blist_body, theta, M)
    print(f"\nautograd 위치 자코비안 (3×6):")
    print(J_pos)

    # 해석적 자코비안의 선속도 부분과 비교
    # 주의: J_b의 v 부분은 body frame 기준이므로 직접 비교는 안 됨
    # space frame FK의 위치에 대한 미분과 비교
    print("\nautograd 전체 자코비안 (16×6) 중 위치 행:")
    J_full = autograd_full_jacobian(Blist_body, theta, M)
    print(J_full[[3, 7, 11], :])  # T[0,3], T[1,3], T[2,3] = position
