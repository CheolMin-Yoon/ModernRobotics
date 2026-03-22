# -*- coding: utf-8 -*-
"""ch06 역기구학 (Inverse Kinematics) - 뉴턴-랩슨 반복법 기반 수치해"""

__all__ = [
    'IKinBody', 'IKinSpace',
]

import numpy as np
import os, sys
from scipy.linalg import expm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ch03_rigid_body_motion.modern_robotics_ch03 import *
from ch04_forward_kinematics.modern_robotics_ch04_ur5e import body_frame_fk, fixed_frame_fk
from ch05_velocity_kinematics.modern_robotics_ch05 import BodyJacobian, SpaceJacobian
from params.ur5e import *  # noqa: F401,F403 — UR5e 파라미터


def IKinBody(Blist_vec, M, T_sd, thetalist0, ew=1e-3, ev=1e-3, max_iter=100):
    """물체꼴(Body frame) 뉴턴-랩슨 역기구학

    알고리즘 (Modern Robotics Algorithm 6.2):
        1. T_bd(θ_i) = T_bs(θ_i)^{-1} * T_sd
        2. V_b = [log(T_bd)]^v   (se3 -> 6-vec twist)
        3. θ_{i+1} = θ_i + J_b^{†}(θ_i) * V_b
        4. ||w_b|| < ew AND ||v_b|| < ev 이면 수렴

    Args:
        Blist_vec: 물체꼴 스크류 축 리스트 (각 6x1)
        M:         영 위치 EE 변환행렬 (4x4)
        T_sd:      목표 EE pose (4x4)
        thetalist0: 초기 관절각 추정값 (nx1)
        ew:        각속도 허용 오차
        ev:        선속도 허용 오차
        max_iter:  최대 반복 횟수

    Returns:
        thetalist: 수렴된 관절각 (nx1)
        success:   수렴 여부 (bool)
    """
    thetalist = np.array(thetalist0, dtype=float)
    Blist_body_se3 = [Vec2se3(B) for B in Blist_vec]

    for i in range(max_iter):
        # 현재 θ에서의 FK
        T_bs = body_frame_fk(Blist_body_se3, thetalist, M)

        # 오차 변환행렬 T_bd = T_bs^{-1} * T_sd
        T_bd = TransInv(T_bs) @ T_sd

        # 트위스트 오차: V_b = [log(T_bd)]^v
        se3mat = MatrixLog(T_bd)
        V_b = se32Vec(se3mat)

        # 수렴 판정
        w_err = np.linalg.norm(V_b[:3])
        v_err = np.linalg.norm(V_b[3:])
        if w_err < ew and v_err < ev:
            return thetalist, True

        # 자코비안 의사역행렬로 관절각 갱신 (순수 뉴턴-랩슨)
        J_b = BodyJacobian(Blist_vec, thetalist)
        thetalist = thetalist + np.linalg.pinv(J_b) @ V_b

    return thetalist, False


def IKinSpace(Slist_vec, M, T_sd, thetalist0, ew=1e-3, ev=1e-3, max_iter=100):
    """공간꼴(Space frame) 뉴턴-랩슨 역기구학

    알고리즘 (Modern Robotics Algorithm 6.1):
        1. T_bs(θ_i) = FK_space(θ_i)
        2. T_bd = T_bs^{-1} * T_sd  ->  V_b = [log(T_bd)]^v
        3. V_s = [Ad_{T_bs}] * V_b
        4. θ_{i+1} = θ_i + J_s^{†}(θ_i) * V_s
        5. ||w_s|| < ew AND ||v_s|| < ev 이면 수렴

    Args:
        Slist_vec: 공간꼴 스크류 축 리스트 (각 6x1)
        M:         영 위치 EE 변환행렬 (4x4)
        T_sd:      목표 EE pose (4x4)
        thetalist0: 초기 관절각 추정값 (nx1)
        ew:        각속도 허용 오차
        ev:        선속도 허용 오차
        max_iter:  최대 반복 횟수

    Returns:
        thetalist: 수렴된 관절각 (nx1)
        success:   수렴 여부 (bool)
    """
    thetalist = np.array(thetalist0, dtype=float)
    Slist_space_se3 = [Vec2se3(S) for S in Slist_vec]

    for i in range(max_iter):
        # 현재 θ에서의 FK (공간꼴)
        T_bs = fixed_frame_fk(Slist_space_se3, thetalist, M)

        # 오차 트위스트 (물체꼴로 구한 뒤 공간꼴로 변환)
        T_bd = TransInv(T_bs) @ T_sd
        V_b = se32Vec(MatrixLog(T_bd))
        V_s = Adjoint(T_bs) @ V_b

        # 수렴 판정
        w_err = np.linalg.norm(V_s[:3])
        v_err = np.linalg.norm(V_s[3:])
        if w_err < ew and v_err < ev:
            return thetalist, True

        # 공간 자코비안 의사역행렬로 갱신
        J_s = SpaceJacobian(Slist_vec, thetalist)
        thetalist = thetalist + np.linalg.pinv(J_s) @ V_s

    return thetalist, False


if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)

    print("=== UR5e 역기구학 (뉴턴-랩슨) ===\n")

    # 목표 pose: θ = [0, -π/2, 0, 0, π/2, 0] 에서의 FK 결과를 목표로 설정
    theta_desired = np.array([0, -np.pi/2, 0, 0, np.pi/2, 0])
    T_sd = body_frame_fk(Blist_e_body, theta_desired, M_e)

    print("목표 T_sd:")
    print(T_sd)
    print()

    # 초기 추정값 (목표에 가까운 값으로 시작)
    theta0 = theta_desired + np.array([0.1, -0.1, 0.1, -0.1, 0.1, -0.1])

    # Body frame IK
    theta_sol_b, success_b = IKinBody(Blist_e_body_vec, M_e, T_sd, theta0)
    print(f"[Body IK] 수렴: {success_b}")
    print(f"  θ 해:  {theta_sol_b}")
    print(f"  목표 θ: {theta_desired}")
    print()

    # Space frame IK
    theta_sol_s, success_s = IKinSpace(Slist_e_space_vec, M_e, T_sd, theta0)
    print(f"[Space IK] 수렴: {success_s}")
    print(f"  θ 해:  {theta_sol_s}")
    print(f"  목표 θ: {theta_desired}")

    # FK 검증
    print("\n=== FK 검증 ===")
    T_check_b = body_frame_fk(Blist_e_body, theta_sol_b, M_e)
    T_check_s = fixed_frame_fk(Slist_e_space, theta_sol_s, M_e)
    print(f"Body IK FK 결과 == T_sd: {np.allclose(T_check_b, T_sd, atol=1e-3)}")
    print(f"Space IK FK 결과 == T_sd: {np.allclose(T_check_s, T_sd, atol=1e-3)}")

    # 영 위치에서 시작하는 테스트도 수행
    print("\n=== 영 위치 초기값 테스트 ===")
    theta0_zero = np.zeros(6)
    theta_sol_b2, success_b2 = IKinBody(Blist_e_body_vec, M_e, T_sd, theta0_zero)
    print(f"[Body IK from zero] 수렴: {success_b2}")
    print(f"  θ 해: {theta_sol_b2}")
    if success_b2:
        T_check = body_frame_fk(Blist_e_body, theta_sol_b2, M_e)
        print(f"  FK 검증: {np.allclose(T_check, T_sd, atol=1e-3)}")
