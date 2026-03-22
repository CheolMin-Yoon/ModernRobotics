# -*- coding: utf-8 -*-
"""UR5 기구학 파라미터 (Modern Robotics 교과서 기준, 단위: mm)"""

import numpy as np
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ch03_rigid_body_motion.modern_robotics_ch03 import *

# ============================================================
# 링크 파라미터 (단위: mm)
# ============================================================
L1 = 425
L2 = 392
W1 = 109
W2 = 82
H1 = 89
H2 = 95

# ============================================================
# 영 위치 EE 변환행렬 M
# ============================================================
R = np.array([[-1, 0, 0],
              [ 0, 0, 1],
              [ 0, 1, 0]])

p = np.array([[L1 + L2],
              [W1 + W2],
              [H1 - H2]])

M = np.block([[R,              p],
              [np.zeros((1, 3)), np.array([[1]])]])

# ============================================================
# 공간꼴 스크류 축
# ============================================================
w1 = np.array([0, 0, 1])
w2 = np.array([0, 1, 0])
w3 = np.array([0, 1, 0])
w4 = np.array([0, 1, 0])
w5 = np.array([0, 0, -1])
w6 = np.array([0, 1, 0])

q1 = np.array([0, 0, 0])
q2 = np.array([0, 0, 0])
q3 = np.array([L1, W1, -H1])
q4 = np.array([L2, 0, 0])
q5 = np.array([0, W1 - W2, 0])
q6 = np.array([0, 0, -H2])

v1 = -np.cross(w1, q1)
v2 = -np.cross(w2, q2)
v3 = -np.cross(w3, q3)
v4 = -np.cross(w4, q4)
v5 = -np.cross(w5, q5)
v6 = -np.cross(w6, q6)

Slist_space_vec = [np.concatenate([w, -np.cross(w, q)]) for w, q in
                   zip([w1, w2, w3, w4, w5, w6],
                       [q1, q2, q3, q4, q5, q6])]

Slist_space = [Vec2se3(S) for S in Slist_space_vec]

M_inv = TransInv(M)
Ad_Minv = Adjoint(M_inv)

Blist_body_vec = [Ad_Minv @ S for S in Slist_space_vec]
Blist_body = [Vec2se3(B) for B in Blist_body_vec]

# URDF 경로
URDF_PATH = os.path.join(os.path.dirname(__file__), '..',
    'urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/'
    'universal_robots/ur_description/urdf/ur5.urdf')
