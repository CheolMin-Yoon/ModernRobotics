# -*- coding: utf-8 -*-
"""ch04 정기구학 (Forward Kinematics) — PyTorch 버전

스크류 이론 기반 PoE (Product of Exponentials) FK.
autograd를 통해 FK의 θ에 대한 미분이 자동으로 가능합니다.
"""

__all__ = [
    'L1', 'L2', 'W1', 'W2', 'H1', 'H2', 'M',
    'thetalist',
    'Slist_space', 'Slist_space_vec',
    'Blist_body', 'Blist_body_vec',
    'body_frame_fk', 'fixed_frame_fk',
]

import torch
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ch03_rigid_body_motion.modern_robotics_ch03_torch import *

# UR5 파라미터 (mm)
L1 = 425
L2 = 392
W1 = 109
W2 = 82
H1 = 89
H2 = 95

_dtype = torch.float64

R = torch.tensor([[-1, 0, 0],
                   [ 0, 0, 1],
                   [ 0, 1, 0]], dtype=_dtype)

p = torch.tensor([[L1 + L2],
                   [W1 + W2],
                   [H1 - H2]], dtype=_dtype)

M = torch.cat([torch.cat([R, p], dim=1),
               torch.tensor([[0, 0, 0, 1]], dtype=_dtype)], dim=0)

thetalist = [0, -torch.pi / 2, 0, 0, torch.pi / 2, 0]

# 공간꼴 스크류 축
_w_space = [
    torch.tensor([0, 0, 1], dtype=_dtype),
    torch.tensor([0, 1, 0], dtype=_dtype),
    torch.tensor([0, 1, 0], dtype=_dtype),
    torch.tensor([0, 1, 0], dtype=_dtype),
    torch.tensor([0, 0, -1], dtype=_dtype),
    torch.tensor([0, 1, 0], dtype=_dtype),
]
_q_space = [
    torch.tensor([0, 0, 0], dtype=_dtype),
    torch.tensor([0, 0, 0], dtype=_dtype),
    torch.tensor([L1, W1, -H1], dtype=_dtype),
    torch.tensor([L2, 0, 0], dtype=_dtype),
    torch.tensor([0, W1 - W2, 0], dtype=_dtype),
    torch.tensor([0, 0, -H2], dtype=_dtype),
]

Slist_space_vec = [torch.cat([w, -torch.linalg.cross(w, q)])
                   for w, q in zip(_w_space, _q_space)]

Slist_space = [Vec2se3(S) for S in Slist_space_vec]

# 물체꼴 스크류: B_i = [Ad_{M^{-1}}] @ S_i
M_inv = TransInv(M)
Ad_Minv = Adjoint(M_inv)

Blist_body_vec = [Ad_Minv @ S for S in Slist_space_vec]
Blist_body = [Vec2se3(B) for B in Blist_body_vec]


# 4.1 물체꼴 FK
def body_frame_fk(Blist_body, thetalist, M=M):
    """물체꼴 PoE 정기구학: T = M @ exp([B1]θ1) @ ... @ exp([Bn]θn)"""
    T = M.clone()
    for B, theta in zip(Blist_body, thetalist):
        T = T @ MatrixExp6(B * theta)
    return T


# 4.2 공간꼴 FK
def fixed_frame_fk(Slist_space, thetalist, M=M):
    """공간꼴 PoE 정기구학: T = exp([S1]θ1) @ ... @ exp([Sn]θn) @ M"""
    T = torch.eye(4, dtype=M.dtype, device=M.device)
    for S, theta in zip(Slist_space, thetalist):
        T = T @ MatrixExp6(S * theta)
    T = T @ M
    return T


if __name__ == '__main__':
    torch.set_printoptions(precision=4, sci_mode=False)
    print("=== ch04 PyTorch FK 테스트 ===\n")

    T_body = body_frame_fk(Blist_body, thetalist)
    T_space = fixed_frame_fk(Slist_space, thetalist)
    print("Body FK:")
    print(T_body)
    print("\nSpace FK:")
    print(T_space)
    print(f"\n일치 여부: {torch.allclose(T_body, T_space, atol=1e-6)}")

    # autograd: FK의 θ에 대한 미분
    theta_grad = torch.tensor(thetalist, dtype=_dtype, requires_grad=True)
    T_grad = body_frame_fk(Blist_body, theta_grad)
    loss = T_grad[:3, 3].sum()  # position의 합
    loss.backward()
    print(f"\n∂(sum p)/∂θ = {theta_grad.grad}")
