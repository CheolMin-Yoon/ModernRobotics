__all__ = [
    'Rot_rpy',
    'spatial_inertia', 'lie_bracket', 'calculate_wrench',
    'transform_to_space',
    'RNEA', 'MassMatrix', 'MassMatrixCRBA', 'VelQuadraticForces', 'GravityForces',
]

import numpy as np
import os, sys
from scipy.linalg import expm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ch03_rigid_body_motion.modern_robotics_ch03 import *
from ch05_velocity_kinematics.modern_robotics_ch05 import *
from params.ur5e import *


def Rot_rpy(r, p, y):
    return MatrixExp3([1, 0, 0], r) @ MatrixExp3([0, 1, 0], p) @ MatrixExp3([0, 0, 1], y)


# 8.2.2 twist-wrench 공식 정리 p404 (단일 강체 기준, 6x6 / 6x1)

# 공간 관성 행렬 6x6
def spatial_inertia(I_b, m):
    G_b = np.block([[np.diag(I_b),      np.zeros((3, 3))],
                    [np.zeros((3, 3)),   m * np.eye(3)]])
    return G_b


# 리 브라켓 6x6
def lie_bracket(V_b):
    w_b = V_b[:3]
    v_b = V_b[3:]
    ad_V = np.block([[Vec2so3(w_b),    np.zeros((3, 3))],
                     [Vec2so3(v_b),    Vec2so3(w_b)]])
    return ad_V


# 렌치 공식 6x1
def calculate_wrench(G_b, dV_b, V_b):
    ad_V = lie_bracket(V_b)
    F_b = G_b @ dV_b - ad_V.T @ G_b @ V_b
    return F_b


# 8.2.3 다른 좌표계에서의 동역학
def transform_to_space(T_sb, G_b, V_b, F_b):
    Ad = Adjoint(T_sb)
    Ad_inv_T = Adjoint(TransInv(T_sb)).T
    G_s = Ad_inv_T @ G_b @ Adjoint(TransInv(T_sb))
    V_s = Ad @ V_b
    F_s = Ad_inv_T @ F_b
    return G_s, V_s, F_s


# 8.3 역 뉴턴 - 오일러 동역학

def RNEA(thetalist, dthetalist, ddthetalist, g, F_tip,
         Mlist, Glist, Slist):
    """8.3 역 뉴턴-오일러 동역학 (MR Algorithm 8.2)

    Args:
        thetalist:   관절 각도       (n,)
        dthetalist:  관절 속도       (n,)
        ddthetalist: 관절 가속도     (n,)
        g:           중력 벡터 (3,)  예: [0, 0, -9.81]
        F_tip:       EE 팁 렌치 (6,)
        Mlist:       인접 링크 변환행렬 [M_01, ..., M_(n-1,n), M_ne]  len=n+1
        Glist:       공간 관성 행렬  [G_1, ..., G_n]  len=n
        Slist:       공간꼴 스크류 축 [S_1, ..., S_n]  len=n

    Returns:
        tau: 관절 토크 (n,)
    """
    n = len(thetalist)

    # ── A_i: 공간꼴 S_i → 링크 i 프레임 기준 스크류 축 ──
    # A_i = Ad_{T_{0,i}^{-1}} @ S_i,  T_{0,i} = M_01 @ ... @ M_{i-1,i}
    Alist = []
    T_0i = np.eye(4)
    for i in range(n):
        T_0i = T_0i @ Mlist[i]
        Ai = Adjoint(TransInv(T_0i)) @ Slist[i]
        Alist.append(Ai)

    # ── 베이스 초기값 ──
    V  = [np.zeros(6)]
    Vd = [np.concatenate([np.zeros(3), -np.array(g)])]

    # ── Forward pass: V_i, Vd_i ──
    T_list = []
    for i in range(n):
        Ai = Alist[i]
        # T_{i,i-1} = exp(-[A_i]*θ_i) @ M_{i-1,i}^{-1}
        Ti = MatrixExp6(Vec2se3(-Ai * thetalist[i])) @ TransInv(Mlist[i])
        T_list.append(Ti)

        AdTi = Adjoint(Ti)
        Vi  = AdTi @ V[i] + Ai * dthetalist[i]
        Vdi = AdTi @ Vd[i] \
              + lie_bracket(Vi) @ Ai * dthetalist[i] \
              + Ai * ddthetalist[i]

        V.append(Vi)
        Vd.append(Vdi)

    # ── Backward pass: F_i, tau_i ──
    F   = np.array(F_tip, dtype=float)
    tau = np.zeros(n)

    for i in range(n - 1, -1, -1):
        Gi  = Glist[i]
        Ai  = Alist[i]
        Vi  = V[i + 1]
        Vdi = Vd[i + 1]

        # 다음 링크 → 현재 링크 렌치 전달
        if i == n - 1:
            T_next = TransInv(Mlist[n])  # M_{n,n+1}^{-1}
        else:
            T_next = T_list[i + 1]

        Fi = Gi @ Vdi - lie_bracket(Vi).T @ Gi @ Vi \
             + Adjoint(T_next).T @ F

        tau[i] = Fi @ Ai
        F = Fi

    return tau

# QP를 위한 유틸리티 함수들

def MassMatrix(thetalist, Mlist, Glist, Slist):
    n = len(thetalist)
    M = np.zeros((n, n))
    
    # 속도와 중력을 0으로 설정
    dthetalist_zero = np.zeros(n)
    g_zero = np.array([0, 0, 0])
    F_tip_zero = np.zeros(6)
    
    for i in range(n):
        # i번째 조인트 가속도만 1로 설정 (Unit acceleration)
        ddthetalist_unit = np.zeros(n)
        ddthetalist_unit[i] = 1
        
        # RNEA를 호출하여 해당 가속도를 만들기 위한 토크(행렬의 열) 산출
        M[:, i] = RNEA(thetalist, dthetalist_zero, ddthetalist_unit, 
                       g_zero, F_tip_zero, Mlist, Glist, Slist)
        
    return M

def MassMatrixCRBA(thetalist, Mlist, Glist, Slist):
    """CRBA (Composite Rigid Body Algorithm) — MR 교재 8.3절 기반

    RNEA를 n번 호출하는 대신, 링크 끝에서 베이스 방향으로
    Composite Inertia를 누적하여 M(θ)를 한 번에 구성.

    복잡도: O(n²)  vs  RNEA 기반 O(n³)
    """
    n = len(thetalist)

    # ── Step 1: A_i, T_{i,i-1} 계산 (RNEA forward pass와 동일) ──
    Alist = []
    T_0i = np.eye(4)
    for i in range(n):
        T_0i = T_0i @ Mlist[i]
        Ai = Adjoint(TransInv(T_0i)) @ Slist[i]
        Alist.append(Ai)

    # T_{i, i-1}: 링크 i 프레임 → 링크 i-1 프레임
    T_list = []
    for i in range(n):
        Ai = Alist[i]
        Ti = MatrixExp6(Vec2se3(-Ai * thetalist[i])) @ TransInv(Mlist[i])
        T_list.append(Ti)

    # ── Step 2: Composite Inertia 초기화 (각 링크 자체 관성) ──
    # IC[i]: 링크 i 프레임 기준 composite spatial inertia
    IC = [G.copy() for G in Glist]

    # ── Step 3: Backward sweep — IC 누적 ──
    # IC[i] += Ad_{T_{i+1,i}}^T @ IC[i+1] @ Ad_{T_{i+1,i}}
    for i in range(n - 2, -1, -1):
        # IC[i+1]을 링크 i 프레임으로 변환하여 누적
        # T_list[i+1] = T_{i+1, i}  →  T_{i, i+1} = TransInv(T_list[i+1])
        Ad = Adjoint(TransInv(T_list[i + 1]))
        IC[i] = IC[i] + Ad.T @ IC[i + 1] @ Ad

    # ── Step 4: M 조립 ──
    M = np.zeros((n, n))
    for i in range(n):
        # 대각 원소
        M[i, i] = Alist[i] @ IC[i] @ Alist[i]

        # 링크 i+1 ~ n-1 방향으로 off-diagonal 채우기
        # F = IC[i] @ A[i] 를 링크 j 프레임으로 전파
        F = IC[i] @ Alist[i]
        for j in range(i + 1, n):
            # F를 링크 j 프레임으로 변환: F_j = Ad_{T_{j,j-1}}^{-T} @ F_{j-1}
            # Ad_{T_{j,j-1}}^{-T} = Ad_{T_{j-1,j}}^T
            F = Adjoint(TransInv(T_list[j])).T @ F
            M[i, j] = Alist[j] @ F
            M[j, i] = M[i, j]  # 대칭

    return M


def VelQuadraticForces(thetalist, dthetalist, Mlist, Glist, Slist):
    n = len(thetalist)
    # 가속도와 중력을 0으로 설정
    ddthetalist_zero = np.zeros(n)
    g_zero = np.array([0, 0, 0])
    F_tip_zero = np.zeros(6)
    
    # RNEA 결과값이 곧 C(theta, dtheta) * dtheta 임
    c_vector = RNEA(thetalist, dthetalist, ddthetalist_zero, 
                    g_zero, F_tip_zero, Mlist, Glist, Slist)
    
    return c_vector
    

def GravityForces(thetalist, g, Mlist, Glist, Slist):
    n = len(thetalist)
    
    # 속도와 가속도를 0으로 설정
    dthetalist_zero = np.zeros(n)
    ddthetalist_zero = np.zeros(n)
    F_tip_zero = np.zeros(6)
    
    # RNEA 호출
    g_vector = RNEA(thetalist, dthetalist_zero, ddthetalist_zero, 
                    g, F_tip_zero, Mlist, Glist, Slist)
    
    return g_vector


if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)

    thetalist = [0, -np.pi/2, 0, 0, np.pi/2, 0]
    dthetalist = np.array([0.1, 0.2, -0.1, 0.05, 0.3, -0.2])
    ddthetalist = np.array([0.01, 0.02, -0.01, 0.005, 0.03, -0.02])

    # 물체 자코비안으로 EE 트위스트 계산
    J_b = BodyJacobian(Blist_body_vec, thetalist)
    V_b = J_b @ dthetalist
    dV_b = J_b @ ddthetalist

    G_b = spatial_inertia(inertia[0], mass[0])
    ad_V = lie_bracket(V_b)
    F_b = calculate_wrench(G_b, dV_b, V_b)

    print("V_b:")
    print(V_b)
    print("\nG_b:")
    print(G_b)
    print("\nad_V:")
    print(ad_V)
    print("\nF_b:")
    print(F_b)
