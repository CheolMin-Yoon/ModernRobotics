__all__ = [
    'L1', 'L2', 'W1', 'W2', 'H1', 'H2', 'M',
    'thetalist',
    'Slist_space', 'Slist_space_vec',
    'Blist_body', 'Blist_body_vec',
    'body_frame_fk', 'fixed_frame_fk',
]

# 4장의 FK를 위한 사전 정의 요소들 

# 1. UR5 6DOF 로봇 파라미터 단위 (mm)
L1 = 425
L2 = 392
W1 = 109
W2 = 82
H1 = 89
H2 = 95

# UR5e 파라미터 (mujoco_menagerie 기준, mm)
# L1 = 425
# L2 = 392
# W1 = 138
# W2 = 127
# H1 = 163
# H2 = 100

# 2. 모든 조인트가 0일 때의 EE 좌표계 변환 행렬 M 
# 책의 222p 6DOF UR5 로봇 기반 M 행렬로 작성

import numpy as np
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ch03_rigid_body_motion.modern_robotics_ch03 import *

R = np.array([[-1, 0, 0],
              [ 0, 0, 1],
              [ 0, 1, 0]])

p = np.array([[L1 + L2],
              [W1 + W2],
              [H1 - H2]])

M = np.block([[R,              p],
             [np.zeros((1, 3)), np.array([[1]])]])

# 3. thetalist -> theta_2: -90, theta_5: 90
thetalist = [0, -np.pi/2, 0, 0, np.pi/2, 0]

# 4. Blist

# 공간꼴 PoE
w1_space = np.array([0, 0, 1])
w2_space = np.array([0, 1, 0])
w3_space = np.array([0, 1, 0])
w4_space = np.array([0, 1, 0])
w5_space = np.array([0, 0, -1])
w6_space = np.array([0, 1, 0])

# 스크류 축 선상의 한점 q
# q1과 q2는 동일한 원점을 공유 가능
q1_space = np.array([0, 0, 0])
q2_space = np.array([0, 0, 0])
q3_space = np.array([L1, W1, -H1])
q4_space = np.array([L2, 0, 0])
q5_space = np.array([0, W1 - W2, 0])
q6_space = np.array([0, 0, -H2])

'''
# 물체꼴 PoE
w1_body = np.array([0, 1, 0])
w2_body = np.array([0, 0, 1])
w3_body = np.array([0, 0, 1])
w4_body = np.array([0, 0, 1])
w5_body = np.array([0, -1, 0])
w6_body = np.array([0, 0, 1])
 
q1_body = np.array([0, 0, -W1])
q2_body = np.array([L1, 0, 0])
q3_body = np.array([L2, 0, 0])
q4_body = np.array([0, H2, 0])
q5_body = np.array([0, 0, -W2])
q6_body = np.array([0, 0, 0])
'''

# v = -w x q (외적)
v1_space = -np.cross(w1_space, q1_space)
v2_space = -np.cross(w2_space, q2_space)
v3_space = -np.cross(w3_space, q3_space)
v4_space = -np.cross(w4_space, q4_space)
v5_space = -np.cross(w5_space, q5_space)
v6_space = -np.cross(w6_space, q6_space)

# 스크류 [S_i] se(3) 4x4 행렬: [[w], v; 0, 0] -> ch03의 Vec2se3 사용

S1_space = Vec2se3(np.concatenate([w1_space, v1_space]))
S2_space = Vec2se3(np.concatenate([w2_space, v2_space]))
S3_space = Vec2se3(np.concatenate([w3_space, v3_space]))
S4_space = Vec2se3(np.concatenate([w4_space, v4_space]))
S5_space = Vec2se3(np.concatenate([w5_space, v5_space]))
S6_space = Vec2se3(np.concatenate([w6_space, v6_space]))

Slist_space = [S1_space, S2_space, S3_space, S4_space, S5_space, S6_space]

# B_i = [Ad_{M^{-1}}] * S_i 로 물체꼴 스크류 계산
M_inv = TransInv(M)
Ad_Minv = Adjoint(M_inv)

Slist_space_vec = [np.concatenate([w, -np.cross(w, q)]) for w, q in
                   zip([w1_space, w2_space, w3_space, w4_space, w5_space, w6_space],
                       [q1_space, q2_space, q3_space, q4_space, q5_space, q6_space])]

Blist_body_vec = [Ad_Minv @ S for S in Slist_space_vec]
Blist_body = [Vec2se3(B) for B in Blist_body_vec]


# forward kinematics의 스크류 운동 기반 해석 (D-H 기반 해석 아님)
# 4.1 영 위치에서의 EE 좌표계 M이 주어지고 EE 좌표계 기준 PoE 해석

# 입력: 관절 스크류 Blist (nx1), 관절 값 thetalist (nx1)
# 출력: EE 좌표계의 pose

from scipy.linalg import expm

def body_frame_fk(Blist_body, thetalist, M=M):
  
    T_body = np.copy(M)
    for B, theta in zip(Blist_body, thetalist):
        T_body = T_body @ expm(B * theta)
        
    return T_body


# 4.2 영 위치에서의 EE 좌표계 M이 주어지고 고정 좌표계 기준 PoE 해석

# 입력: 관절 스크류 Slist (nx1), 관절 값 thetalist (nx1)
# 출력: EE pose

def fixed_frame_fk(Slist_space, thetalist, M=M):
   
    T_space = np.eye(4)
    for S, theta in zip(Slist_space, thetalist):
        T_space = T_space @ expm(S * theta)
    T_space = T_space @ M
    
    return T_space


if __name__ == '__main__':
    T_body = body_frame_fk(Blist_body, thetalist)
    T_space = fixed_frame_fk(Slist_space, thetalist)
    print(T_body)
    print(T_space)