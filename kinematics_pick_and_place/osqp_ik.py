# -*- coding: utf-8 -*-
"""라그랑주 승수 기반 역기구학 (OSQP Constrained Optimization IK)

운동에너지를 최소화하면서 기구학 구속 + 관절 제한을 만족하는 해를 구한다.

매 반복마다 선형화된 QP:
    min  0.5 * dθ^T * M(θ) * dθ
    s.t. J_b * dθ = V_b                (기구학 오차 트위스트)
         q_lower - θ <= dθ <= q_upper - θ  (관절 위치 제한)
         -dq_max*dt  <= dθ <= dq_max*dt    (관절 속도 제한)

OSQP 표준형:
    min  0.5 x^T P x + q^T x
    s.t. l <= A x <= u

    P = M(θ),  q = 0
    A = [J_b; I],  등식 + 부등식 구속 결합
"""

import numpy as np
import osqp
from scipy import sparse
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ch03_rigid_body_motion.modern_robotics_ch03 import *
from ch04_forward_kinematics.modern_robotics_ch04_ur5e import body_frame_fk
from ch05_velocity_kinematics.modern_robotics_ch05 import BodyJacobian
from ch08_dynamics.modern_robotics_ch08 import MassMatrix
from params.ur5e import (
    Blist_body, Blist_body_vec, M_e,
    Slist_space_vec, Mlist, Glist,
    q_lower, q_upper, dq_max, tau_max,
)


def osqp_ik(Blist_vec, M, T_sd, thetalist0,
                  use_mass_matrix=True,
                  ew=1e-3, ev=1e-3, max_iter=100,
                  alpha=1.0, dt=0.1):
    """라그랑주 승수 기반 역기구학 (OSQP QP solver)

    매 반복:
        1. FK → T_bs, 오차 T_bd = T_bs^{-1} T_sd
        2. V_b = se32Vec(MatrixLog(T_bd))
        3. J_b = BodyJacobian(θ)
        4. QP 풀기:
            min  0.5 dθ^T M(θ) dθ
            s.t. J_b dθ = V_b                     (등식: 기구학)
                 q_lower - θ <= dθ <= q_upper - θ  (부등식: 위치 제한)
                 -dq_max*dt  <= dθ <= dq_max*dt    (부등식: 속도 제한)
        5. θ ← θ + α * dθ

    Args:
        Blist_vec:       물체꼴 스크류 축 리스트 (각 6-vec)
        M:               영 위치 EE 변환행렬 (4x4)
        T_sd:            목표 EE pose (4x4)
        thetalist0:      초기 관절각 추정값 (n,)
        use_mass_matrix: True면 M(θ) 사용, False면 단위행렬
        ew:              각속도 허용 오차
        ev:              선속도 허용 오차
        max_iter:        최대 반복 횟수
        alpha:           스텝 크기 (0 < α ≤ 1)
        dt:              속도 제한 시간 스텝 [s]

    Returns:
        thetalist: 수렴된 관절각 (n,)
        success:   수렴 여부 (bool)
    """
    thetalist = np.array(thetalist0, dtype=float)
    n = len(thetalist)
    Blist_se3 = [Vec2se3(B) for B in Blist_vec]

    for it in range(max_iter):
        # 1. FK → 오차 트위스트
        T_bs = body_frame_fk(Blist_se3, thetalist, M)
        T_bd = TransInv(T_bs) @ T_sd
        V_b = se32Vec(MatrixLog(T_bd))

        # 2. 수렴 판정
        w_err = np.linalg.norm(V_b[:3])
        v_err = np.linalg.norm(V_b[3:])
        if w_err < ew and v_err < ev:
            return thetalist, True

        # 3. 자코비안
        J_b = BodyJacobian(Blist_vec, thetalist)

        # 4. 목적함수 P
        if use_mass_matrix:
            M_theta = MassMatrix(thetalist, Mlist, Glist, Slist_space_vec)
            M_theta = 0.5 * (M_theta + M_theta.T)
            M_theta += 1e-8 * np.eye(n)
        else:
            M_theta = np.eye(n)

        # 5. 구속 조건 조립
        #    A = [J_b (6×n)]   등식: l = u = V_b
        #        [I_n (n×n)]   부등식: 위치+속도 제한의 교집합
        I_n = np.eye(n)
        A = np.vstack([J_b, I_n])

        # 부등식 경계: 위치 제한과 속도 제한의 교집합
        dq_pos_lower = q_lower - thetalist   # 위치 하한까지 남은 여유
        dq_pos_upper = q_upper - thetalist   # 위치 상한까지 남은 여유
        dq_vel = dq_max * dt                 # 속도 제한에 의한 최대 변위

        box_lower = np.maximum(dq_pos_lower, -dq_vel)
        box_upper = np.minimum(dq_pos_upper,  dq_vel)

        # 전체 l, u 벡터
        l = np.concatenate([V_b, box_lower])
        u = np.concatenate([V_b, box_upper])

        # 6. OSQP 풀기
        P_sp = sparse.csc_matrix(M_theta)
        A_sp = sparse.csc_matrix(A)
        q_vec = np.zeros(n)

        solver = osqp.OSQP()
        solver.setup(P_sp, q_vec, A_sp, l, u,
                     verbose=False,
                     eps_abs=1e-8, eps_rel=1e-8,
                     max_iter=4000, polish=True)
        result = solver.solve()

        if result.info.status == 'solved' or result.info.status == 'solved_inaccurate':
            dtheta = result.x
        else:
            # fallback: 의사역행렬 (구속 무시)
            dtheta = np.linalg.pinv(J_b) @ V_b
            # 클리핑
            dtheta = np.clip(dtheta, box_lower, box_upper)

        # 7. 갱신
        thetalist = thetalist + alpha * dtheta

    return thetalist, False


if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)

    print("=== UR5e 라그랑주 승수 IK (OSQP + Joint Limits) ===\n")

    # 목표: 알려진 θ에서의 FK 결과를 목표 pose로 설정
    theta_desired = np.array([0, -np.pi/2, 0, 0, np.pi/2, 0])
    T_sd = body_frame_fk(Blist_body, theta_desired, M_e)

    print("목표 T_sd:")
    print(T_sd)
    print(f"관절 위치 제한: lower={np.rad2deg(q_lower)}, upper={np.rad2deg(q_upper)}")
    print(f"관절 속도 제한: {np.rad2deg(dq_max)} deg/s")
    print(f"토크 제한:      {tau_max} Nm")
    print()

    # 초기 추정값 (약간 벗어난 값)
    theta0 = theta_desired + np.array([0.1, -0.1, 0.1, -0.1, 0.1, -0.1])

    # M(θ) + joint limits
    theta_sol, success = osqp_ik(Blist_body_vec, M_e, T_sd, theta0,
                                       use_mass_matrix=True)
    print(f"[M(θ) + limits] 수렴: {success}")
    print(f"  θ 해:  {theta_sol}")
    print(f"  목표 θ: {theta_desired}")
    T_check = body_frame_fk(Blist_body, theta_sol, M_e)
    print(f"  FK 오차: {np.linalg.norm(T_check - T_sd):.6e}")
    print(f"  위치 제한 만족: {np.all(theta_sol >= q_lower) and np.all(theta_sol <= q_upper)}")
    print()

    # 단위행렬 + joint limits (비교)
    theta_sol2, success2 = osqp_ik(Blist_body_vec, M_e, T_sd, theta0,
                                          use_mass_matrix=False)
    print(f"[I + limits]    수렴: {success2}")
    print(f"  θ 해:  {theta_sol2}")
    T_check2 = body_frame_fk(Blist_body, theta_sol2, M_e)
    print(f"  FK 오차: {np.linalg.norm(T_check2 - T_sd):.6e}")
    print()

    # 영 위치에서 시작
    theta0_zero = np.zeros(6)
    theta_sol3, success3 = osqp_ik(Blist_body_vec, M_e, T_sd, theta0_zero,
                                          use_mass_matrix=True)
    print(f"[영위치 시작]   수렴: {success3}")
    print(f"  θ 해:  {theta_sol3}")
    if success3:
        T_c = body_frame_fk(Blist_body, theta_sol3, M_e)
        print(f"  FK 검증: {np.allclose(T_c, T_sd, atol=1e-3)}")
        print(f"  위치 제한 만족: {np.all(theta_sol3 >= q_lower) and np.all(theta_sol3 <= q_upper)}")
