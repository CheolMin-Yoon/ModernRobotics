__all__ = [
    'L1_e', 'L2_e', 'W1_e', 'W2_e', 'H1_e', 'H2_e',
    'Rot_rpy',
    'w1_e', 'w2_e', 'w3_e', 'w4_e', 'w5_e', 'w6_e',
    'q1_e', 'q2_e', 'q3_e', 'q4_e', 'q5_e', 'q6_e',
    'v1_e', 'v2_e', 'v3_e', 'v4_e', 'v5_e', 'v6_e',
    'Slist_e_vec', 'Slist_e', 'Blist_e_vec', 'Blist_e',
    'M_e', 'R_e', 'p_e',
    'm', 'r', 'I_b',
    'spatial_inertia', 'lie_bracket', 'calculate_wrench',
    'transform_to_space',
    'RNEA',
]

import numpy as np
import os, sys
from scipy.linalg import expm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ch03_rigid_body_motion.modern_robotics_ch03 import *
from ch04_forward_kinematics.modern_robotics_ch04 import *
from ch05_velocity_kinematics.modern_robotics_ch05 import *


# ============================================================
# UR5e 로봇 파라미터 (ros-industrial URDF 기준, 단위: m)
# ============================================================

L1_e = 0.425     
L2_e = 0.3922    
W1_e = 0.1333    
W2_e = 0.0997    
H1_e = 0.1625    
H2_e = 0.0996    

# ============================================================
# 공간꼴 스크류 축 (zero config, world frame 기준)
# ============================================================

local_axis = np.array([0, 0, 1])

def Rot_rpy(r, p, y):
    return MatrixExp3([1, 0, 0], r) @ MatrixExp3([0, 1, 0], p) @ MatrixExp3([0, 0, 1], y)

R_w1 = Rot_rpy(0, 0, 0)                                         
R_w2 = R_w1 @ Rot_rpy(np.pi/2, 0, 0)                           
R_w3 = R_w2 @ Rot_rpy(0, 0, 0)                                  
R_w4 = R_w3 @ Rot_rpy(0, 0, 0)                                  
R_w5 = R_w4 @ Rot_rpy(np.pi/2, 0, 0)                          
R_w6 = R_w5 @ Rot_rpy(np.pi/2, np.pi, np.pi)                  

w1_e = R_w1 @ local_axis
w2_e = R_w2 @ local_axis
w3_e = R_w3 @ local_axis
w4_e = R_w4 @ local_axis
w5_e = R_w5 @ local_axis
w6_e = R_w6 @ local_axis

# 스크류 축 선상의 한 점 q (zero config에서 각 조인트의 world frame 위치)
q1_e = np.array([0, 0, H1_e])
q2_e = np.array([0, 0, H1_e])
q3_e = np.array([-L1_e, 0, H1_e])
q4_e = np.array([-(L1_e + L2_e), 0, H1_e])
q5_e = np.array([-(L1_e + L2_e), -W1_e, H1_e])
q6_e = np.array([-(L1_e + L2_e), -(W1_e - H2_e), H1_e])

# v = -[w] @ q  (skew-symmetric 행렬곱)
v1_e = -Vec2so3(w1_e) @ q1_e
v2_e = -Vec2so3(w2_e) @ q2_e
v3_e = -Vec2so3(w3_e) @ q3_e
v4_e = -Vec2so3(w4_e) @ q4_e
v5_e = -Vec2so3(w5_e) @ q5_e
v6_e = -Vec2so3(w6_e) @ q6_e

# 공간꼴 스크류 벡터 S_i = [w_i; v_i] (6x1)
Slist_e_vec = [np.concatenate([w, v]) for w, v in
               zip([w1_e, w2_e, w3_e, w4_e, w5_e, w6_e],
                   [v1_e, v2_e, v3_e, v4_e, v5_e, v6_e])]

# se(3) 행렬 형태
Slist_e = [Vec2se3(S) for S in Slist_e_vec]

# ============================================================
# EE configuration M (zero config에서의 EE 변환행렬)
# ============================================================

# zero config에서 EE(tool0) 위치
p_e = np.array([[-(L1_e + L2_e)],
                [-(W1_e - H2_e + W2_e)],
                [H1_e]])

# zero config에서 EE 회전 (URDF flange→tool0 누적)
R_e = np.array([[-1, 0,  0],
                [ 0, 0,  1],
                [ 0, 1,  0]])

M_e = np.block([[R_e,              p_e],
                [np.zeros((1, 3)), np.array([[1]])]])

# 물체꼴 스크류 B_i = [Ad_{M^{-1}}] * S_i
M_e_inv = TransInv(M_e)
Ad_Me_inv = Adjoint(M_e_inv)

Blist_e_vec = [Ad_Me_inv @ S for S in Slist_e_vec]
Blist_e = [Vec2se3(B) for B in Blist_e_vec]

# ============================================================
# UR5e 동역학 파라미터 (URDF inertial 기준, SI 단위)
# ============================================================

# 각 링크의 질량 m_i (kg)
m = [3.7, 8.393, 2.275, 1.219, 1.219, 0.1879]


# 각 링크의 CoM (링크 로컬 프레임 기준, URDF inertial origin xyz)
r = [
    np.array([0, 0, 0]),          
    np.array([-0.2125, 0, 0.138]),  
    np.array([-0.1961, 0, 0.007]),  
    np.array([0, 0, 0]),          
    np.array([0, 0, 0]),            
    np.array([0, 0, -0.0229]),     
]

# 각 링크의 관성 텐서 (조인트 프레임 기준, URDF inertial origin rpy 반영)
# URDF에서 CoM의 rpy가 있는 링크는 주축이 회전되어 있으므로
# Pinocchio와 일치시키려면 rpy 회전을 반영해야 함
# upper_arm_link: rpy=(0, π/2, 0) → Ixx↔Izz 교환
# forearm_link:   rpy=(0, π/2, 0) → Ixx↔Izz 교환
# [Ixx, Iyy, Izz] → 조인트 프레임 기준
I_b = [
    np.array([0.010267495893, 0.010267495893, 0.00666]),      # shoulder_link (rpy=0)
    np.array([0.0151074, 0.1338857818, 0.1338857818]),        # upper_arm_link (rpy Y=π/2: Ixx↔Izz)
    np.array([0.004095, 0.0312093551, 0.0312093551]),         # forearm_link (rpy Y=π/2: Ixx↔Izz)
    np.array([0.0025598990, 0.0025598990, 0.0021942]),        # wrist_1_link (rpy=0)
    np.array([0.0025598990, 0.0025598990, 0.0021942]),        # wrist_2_link (rpy=0)
    np.array([9.890410e-05, 9.890410e-05, 1.321172e-04]),     # wrist_3_link (rpy=0)
]

# 8.2.2 twist-wrench 공식 정리 p404 (단일 강체 기준, 6x6 / 6x1)

# 공간 관성 행렬 6x6
def spatial_inertia(I_b, m):

    G_b = np.block([[np.diag(I_b),         np.zeros((3, 3))],
                    [np.zeros((3, 3)),        m * np.eye(3)]])
    
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

# 물체 좌표계 기준에서 공간 좌표계 기준으로 
# 슈타이너 정리의 일반화?

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
    """8.3 역 뉴턴-오일러 동역학 (미구현 스켈레톤)
    
    Args:
        thetalist:   관절 각도 (nx1)
        dthetalist:  관절 속도 (nx1)
        ddthetalist: 관절 가속도 (nx1)
        g:           중력 벡터 (3x1)
        F_tip:       EE 팁 렌치 (6x1)
        Mlist:       인접 링크 변환행렬 리스트 [M_{0,1}, ..., M_{n,n+1}]
        Glist:       공간 관성 행렬 리스트 [G_1, ..., G_n]
        Slist:       스크류 축 리스트 [S_1, ..., S_n]
    
    Returns:
        tau: 관절 토크 (nx1)
    """
    # TODO: 구현 예정
    pass

if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)

    dthetalist = np.array([0.1, 0.2, -0.1, 0.05, 0.3, -0.2])
    ddthetalist = np.array([0.01, 0.02, -0.01, 0.005, 0.03, -0.02])

    # 물체 자코비안으로 EE 트위스트 계산
    J_b = BodyJacobian(Blist_e_vec, thetalist)
    V_b = J_b @ dthetalist          # 6x1 물체 트위스트
    dV_b = J_b @ ddthetalist        # 6x1 트위스트 미분

    G_b = spatial_inertia(I_b[0], m[0])
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
