# -*- coding: utf-8 -*-
"""ch03 강체 운동 (Rigid Body Motion) — PyTorch 버전

numpy 기반 modern_robotics_ch03.py의 PyTorch 포팅.
모든 연산이 torch.Tensor 기반이므로 autograd 자동 미분이 가능합니다.
"""

__all__ = [
    'RotInv', 'Vec2so3', 'so32Vec', 'AxisAng3',
    'MatrixExp3', 'MatrixLog3', 'Rp2Trans', 'Trans2Rp',
    'TransInv', 'Vec2se3', 'se32Vec', 'Adjoint',
    'Screw2Axis', 'AxisAng', 'MatrixExp6', 'MatrixLog',
]

import torch


def RotInv(R):
    """3.1 회전 행렬 R의 역행렬 (= 전치)"""
    return R.T


def Vec2so3(omega):
    """3.2 3차원 벡터 → so(3) 반대칭 행렬"""
    omega = omega.flatten()
    w1, w2, w3 = omega[0], omega[1], omega[2]
    zero = torch.zeros(1, dtype=omega.dtype, device=omega.device).squeeze()
    return torch.stack([
        torch.stack([zero, -w3,  w2]),
        torch.stack([ w3, zero, -w1]),
        torch.stack([-w2,  w1, zero]),
    ])


def so32Vec(so3mat):
    """3.3 so(3) 반대칭 행렬 → 3차원 벡터"""
    return torch.stack([so3mat[2, 1], so3mat[0, 2], so3mat[1, 0]])


def AxisAng3(expc3):
    """3.4 지수 좌표 → (단위 회전축, 회전각)"""
    theta = torch.norm(expc3)
    if theta < 1e-6:
        return torch.zeros(3, dtype=expc3.dtype, device=expc3.device), theta
    return expc3 / theta, theta


def MatrixExp3(hat_omega, theta):
    """3.5 로드리게스 공식: so(3) → SO(3)"""
    I = torch.eye(3, dtype=hat_omega.dtype, device=hat_omega.device)
    w_mat = Vec2so3(hat_omega)
    return I + torch.sin(theta) * w_mat + (1 - torch.cos(theta)) * (w_mat @ w_mat)


def MatrixLog3(R):
    """3.6 SO(3) → so(3) 행렬 로그"""
    cos_theta = torch.clamp((torch.trace(R) - 1) / 2.0, -1.0, 1.0)
    theta = torch.acos(cos_theta)

    if theta < 1e-6:
        return torch.zeros(3, 3, dtype=R.dtype, device=R.device)
    elif abs(theta.item() - torch.pi) < 1e-6:
        RpI = R + torch.eye(3, dtype=R.dtype, device=R.device)
        col = torch.argmax(torch.norm(RpI, dim=0))
        w = RpI[:, col] / torch.norm(RpI[:, col])
        return Vec2so3(w * theta)
    else:
        return theta / (2 * torch.sin(theta)) * (R - R.T)


def Rp2Trans(R, p):
    """3.7 (R, p) → 동차 변환행렬 T (4×4)"""
    p = p.reshape(3, 1)
    bottom = torch.zeros(1, 4, dtype=R.dtype, device=R.device)
    bottom[0, 3] = 1.0
    top = torch.cat([R, p], dim=1)
    return torch.cat([top, bottom], dim=0)


def Trans2Rp(T):
    """3.8 T → (R, p)"""
    return T[:3, :3], T[:3, 3]


def TransInv(T):
    """3.9 동차 변환행렬의 역행렬"""
    R, p = Trans2Rp(T)
    Rt = RotInv(R)
    p_new = -(Rt @ p.reshape(3, 1))
    bottom = torch.zeros(1, 4, dtype=T.dtype, device=T.device)
    bottom[0, 3] = 1.0
    top = torch.cat([Rt, p_new], dim=1)
    return torch.cat([top, bottom], dim=0)


def Vec2se3(V):
    """3.10 6차원 twist → se(3) 4×4 행렬"""
    w = V[:3]
    v = V[3:]
    so3 = Vec2so3(w)
    top = torch.cat([so3, v.reshape(3, 1)], dim=1)
    bottom = torch.zeros(1, 4, dtype=V.dtype, device=V.device)
    return torch.cat([top, bottom], dim=0)


def se32Vec(se3mat):
    """3.11 se(3) 4×4 → 6차원 twist"""
    w = so32Vec(se3mat[:3, :3])
    v = se3mat[:3, 3]
    return torch.cat([w, v])


def Adjoint(T):
    """3.12 동차 변환행렬의 6×6 수반 표현 [Ad_T]"""
    R, p = Trans2Rp(T)
    p_mat = Vec2so3(p)
    zeros = torch.zeros(3, 3, dtype=T.dtype, device=T.device)
    top = torch.cat([R, zeros], dim=1)
    bot = torch.cat([p_mat @ R, R], dim=1)
    return torch.cat([top, bot], dim=0)


def Screw2Axis(q, s, h):
    """3.13 (q, s, h) → 정규화된 스크류 축 S (6,)"""
    w = s.flatten()
    v = -torch.linalg.cross(q.flatten(), s.flatten()) + h * w
    return torch.cat([w, v])


def AxisAng(expc6):
    """3.14 6차원 지수 좌표 → (S, theta)"""
    w = expc6[:3]
    v = expc6[3:]
    theta = torch.norm(w)
    if theta.item() == 0:
        theta = torch.norm(v)
    S = expc6 / theta
    return S, theta


def MatrixExp6(se3mat):
    """3.15 se(3) → SE(3) 행렬 지수"""
    so3mat = se3mat[:3, :3]
    w = so32Vec(so3mat)
    v = se3mat[:3, 3].reshape(3, 1)
    I = torch.eye(3, dtype=se3mat.dtype, device=se3mat.device)
    theta = torch.norm(w)

    if theta < 1e-6:
        top = torch.cat([I, v], dim=1)
        bottom = torch.zeros(1, 4, dtype=se3mat.dtype, device=se3mat.device)
        bottom[0, 3] = 1.0
        return torch.cat([top, bottom], dim=0)

    hat_w = w / theta
    w_mat = Vec2so3(hat_w)
    R = MatrixExp3(hat_w, theta)
    G = I * theta + (1 - torch.cos(theta)) * w_mat + (theta - torch.sin(theta)) * (w_mat @ w_mat)
    p = G @ (v / theta)

    top = torch.cat([R, p], dim=1)
    bottom = torch.zeros(1, 4, dtype=se3mat.dtype, device=se3mat.device)
    bottom[0, 3] = 1.0
    return torch.cat([top, bottom], dim=0)


def MatrixLog(T):
    """3.16 SE(3) → se(3) 행렬 로그"""
    R, p = Trans2Rp(T)
    p = p.reshape(3, 1)
    cos_theta = torch.clamp((torch.trace(R) - 1) / 2.0, -1.0, 1.0)
    theta = torch.acos(cos_theta)

    if theta < 1e-6:
        zeros33 = torch.zeros(3, 3, dtype=T.dtype, device=T.device)
        top = torch.cat([zeros33, p], dim=1)
        bottom = torch.zeros(1, 4, dtype=T.dtype, device=T.device)
        return torch.cat([top, bottom], dim=0)

    so3mat = MatrixLog3(R)
    w_mat = so3mat / theta
    I = torch.eye(3, dtype=T.dtype, device=T.device)

    if theta < 1e-3:
        G_inv = (1.0 / theta) * I - 0.5 * w_mat + (theta / 12.0) * (w_mat @ w_mat)
    else:
        G_inv = (1.0 / theta) * I \
                - 0.5 * w_mat \
                + (1.0 / theta - 0.5 / torch.tan(theta / 2.0)) * (w_mat @ w_mat)

    v = theta * (G_inv @ p)
    top = torch.cat([so3mat, v], dim=1)
    bottom = torch.zeros(1, 4, dtype=T.dtype, device=T.device)
    return torch.cat([top, bottom], dim=0)


# ============================================================
# 테스트
# ============================================================
if __name__ == '__main__':
    torch.set_printoptions(precision=4, sci_mode=False)
    print("=== ch03 PyTorch 버전 테스트 ===\n")

    # SO(3) 테스트
    omega = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    theta = torch.tensor(torch.pi / 4, dtype=torch.float64)
    R = MatrixExp3(omega, theta)
    print("MatrixExp3 (z축 45도 회전):")
    print(R)

    so3 = MatrixLog3(R)
    print("\nMatrixLog3:")
    print(so3)

    # SE(3) 테스트
    V = torch.tensor([0.0, 0.0, 1.0, 0.0, 2.0, 0.0], dtype=torch.float64)
    se3 = Vec2se3(V * theta)
    T = MatrixExp6(se3)
    print("\nMatrixExp6:")
    print(T)

    se3_log = MatrixLog(T)
    print("\nMatrixLog:")
    print(se3_log)

    # autograd 테스트
    theta_grad = torch.tensor(torch.pi / 4, dtype=torch.float64, requires_grad=True)
    R_grad = MatrixExp3(omega, theta_grad)
    loss = R_grad.sum()
    loss.backward()
    print(f"\n∂(sum R)/∂θ = {theta_grad.grad:.4f}")
