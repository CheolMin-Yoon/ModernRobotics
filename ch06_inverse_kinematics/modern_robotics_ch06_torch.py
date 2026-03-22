# -*- coding: utf-8 -*-
"""ch06 역기구학 (Inverse Kinematics) — PyTorch 버전

뉴턴-랩슨 반복법 기반 수치해 + autograd 기반 IK.
autograd를 활용하면 해석적 자코비안 없이도 FK 미분만으로 IK를 풀 수 있습니다.
"""

__all__ = [
    'IKinBody', 'IKinSpace', 'IKinAutograd',
]

import torch
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ch03_rigid_body_motion.modern_robotics_ch03_torch import *
from ch05_velocity_kinematics.modern_robotics_ch05_torch import BodyJacobian, SpaceJacobian


def _body_frame_fk(Blist_body_se3, thetalist, M):
    """내부용 FK (se3 행렬 리스트 입력)"""
    T = M.clone()
    for B, theta in zip(Blist_body_se3, thetalist):
        T = T @ MatrixExp6(B * theta)
    return T


def _fixed_frame_fk(Slist_space_se3, thetalist, M):
    """내부용 FK (se3 행렬 리스트 입력)"""
    T = torch.eye(4, dtype=M.dtype, device=M.device)
    for S, theta in zip(Slist_space_se3, thetalist):
        T = T @ MatrixExp6(S * theta)
    return T @ M


def IKinBody(Blist_vec, M, T_sd, thetalist0, ew=1e-3, ev=1e-3, max_iter=100):
    """물체꼴 뉴턴-랩슨 역기구학

    Args:
        Blist_vec: 물체꼴 스크류 축 리스트 (각 6-vec, torch.Tensor)
        M:         영 위치 EE 변환행렬 (4×4)
        T_sd:      목표 EE pose (4×4)
        thetalist0: 초기 관절각 추정값
        ew, ev:    수렴 허용 오차
        max_iter:  최대 반복 횟수

    Returns:
        thetalist, success
    """
    thetalist = thetalist0.clone().detach().to(dtype=torch.float64)
    Blist_se3 = [Vec2se3(B) for B in Blist_vec]

    for i in range(max_iter):
        T_bs = _body_frame_fk(Blist_se3, thetalist, M)
        T_bd = TransInv(T_bs) @ T_sd
        se3mat = MatrixLog(T_bd)
        V_b = se32Vec(se3mat)

        w_err = torch.norm(V_b[:3]).item()
        v_err = torch.norm(V_b[3:]).item()
        if w_err < ew and v_err < ev:
            return thetalist, True

        J_b = BodyJacobian(Blist_vec, thetalist)
        thetalist = thetalist + torch.linalg.pinv(J_b) @ V_b

    return thetalist, False


def IKinSpace(Slist_vec, M, T_sd, thetalist0, ew=1e-3, ev=1e-3, max_iter=100):
    """공간꼴 뉴턴-랩슨 역기구학"""
    thetalist = thetalist0.clone().detach().to(dtype=torch.float64)
    Slist_se3 = [Vec2se3(S) for S in Slist_vec]

    for i in range(max_iter):
        T_bs = _fixed_frame_fk(Slist_se3, thetalist, M)
        T_bd = TransInv(T_bs) @ T_sd
        V_b = se32Vec(MatrixLog(T_bd))
        V_s = Adjoint(T_bs) @ V_b

        w_err = torch.norm(V_s[:3]).item()
        v_err = torch.norm(V_s[3:]).item()
        if w_err < ew and v_err < ev:
            return thetalist, True

        J_s = SpaceJacobian(Slist_vec, thetalist)
        thetalist = thetalist + torch.linalg.pinv(J_s) @ V_s

    return thetalist, False


def IKinAutograd(Blist_body_se3, M, T_sd, thetalist0,
                 lr=0.01, ew=1e-3, ev=1e-3, max_iter=1000):
    """autograd 기반 역기구학 (경사 하강법)

    FK의 출력과 목표 pose 사이의 오차를 loss로 정의하고,
    torch.autograd로 θ에 대한 기울기를 구해 최적화합니다.

    해석적 자코비안 없이 FK 함수만 있으면 IK를 풀 수 있는 것이 장점입니다.

    Args:
        Blist_body_se3: 물체꼴 스크류 se3 행렬 리스트
        M:              영 위치 EE 변환행렬 (4×4)
        T_sd:           목표 EE pose (4×4)
        thetalist0:     초기 관절각
        lr:             학습률
        ew, ev:         수렴 허용 오차
        max_iter:       최대 반복 횟수

    Returns:
        thetalist, success
    """
    theta = thetalist0.clone().detach().to(dtype=torch.float64).requires_grad_(True)
    optimizer = torch.optim.Adam([theta], lr=lr)

    T_sd_detach = T_sd.detach()

    for i in range(max_iter):
        optimizer.zero_grad()

        T_cur = _body_frame_fk(Blist_body_se3, theta, M)

        # 위치 오차
        p_err = T_cur[:3, 3] - T_sd_detach[:3, 3]
        # 자세 오차: R_err = R_cur @ R_sd^T, log로 각도 추출
        R_err = T_cur[:3, :3] @ T_sd_detach[:3, :3].T
        so3_err = MatrixLog3(R_err)
        w_err_vec = so32Vec(so3_err)

        loss_v = torch.sum(p_err ** 2)
        loss_w = torch.sum(w_err_vec ** 2)
        loss = loss_v + loss_w

        # 수렴 판정
        with torch.no_grad():
            if torch.sqrt(loss_w).item() < ew and torch.sqrt(loss_v).item() < ev:
                return theta.detach(), True

        loss.backward()
        optimizer.step()

    return theta.detach(), False


if __name__ == '__main__':
    import numpy as np

    torch.set_printoptions(precision=4, sci_mode=False)
    _dtype = torch.float64

    print("=== ch06 PyTorch 역기구학 테스트 ===\n")

    # UR5e 파라미터를 numpy에서 torch로 변환
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from params.ur5e import (
        M_e as M_np, Blist_body_vec as Bb_np, Blist_body as Bb_se3_np,
        Slist_space_vec as Ss_np, thetalist as th_np,
    )

    M_t = torch.tensor(M_np, dtype=_dtype)
    Bb_vec_t = [torch.tensor(b, dtype=_dtype) for b in Bb_np]
    Bb_se3_t = [torch.tensor(b, dtype=_dtype) for b in Bb_se3_np]
    Ss_vec_t = [torch.tensor(s, dtype=_dtype) for s in Ss_np]

    # 목표 pose: 알려진 θ에서의 FK 결과
    theta_desired = torch.tensor([0, -torch.pi/2, 0, 0, torch.pi/2, 0], dtype=_dtype)
    T_sd = _body_frame_fk(Bb_se3_t, theta_desired, M_t)
    print("목표 T_sd:")
    print(T_sd)

    # 초기 추정값
    theta0 = theta_desired + torch.tensor([0.1, -0.1, 0.1, -0.1, 0.1, -0.1], dtype=_dtype)

    # Body IK
    theta_b, ok_b = IKinBody(Bb_vec_t, M_t, T_sd, theta0)
    print(f"\n[Body IK] 수렴: {ok_b}")
    print(f"  θ 해:  {theta_b}")
    print(f"  목표 θ: {theta_desired}")

    # Space IK
    theta_s, ok_s = IKinSpace(Ss_vec_t, M_t, T_sd, theta0)
    print(f"\n[Space IK] 수렴: {ok_s}")
    print(f"  θ 해:  {theta_s}")

    # Autograd IK
    theta_ag, ok_ag = IKinAutograd(Bb_se3_t, M_t, T_sd, theta0, lr=0.05, max_iter=2000)
    print(f"\n[Autograd IK] 수렴: {ok_ag}")
    print(f"  θ 해:  {theta_ag}")

    # FK 검증
    print("\n=== FK 검증 ===")
    T_check_b = _body_frame_fk(Bb_se3_t, theta_b, M_t)
    T_check_ag = _body_frame_fk(Bb_se3_t, theta_ag, M_t)
    print(f"Body IK FK == T_sd: {torch.allclose(T_check_b, T_sd, atol=1e-3)}")
    print(f"Autograd IK FK == T_sd: {torch.allclose(T_check_ag, T_sd, atol=1e-2)}")
