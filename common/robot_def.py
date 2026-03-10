"""
로봇 정의 — 모든 챕터에서 공유

3DOF RRR 매니퓰레이터를 기본 실습 로봇으로 사용.
PoE (Product of Exponentials) 파라미터와 DH 파라미터 모두 정의.
"""

import numpy as np


class ThreeLinkRobot:
    """3DOF RRR Planar-ish Manipulator"""

    def __init__(self, L1=0.3, L2=0.25, L3=0.2):
        self.n_dof = 3
        self.L1 = L1  # base 높이 (d1)
        self.L2 = L2  # link 2 길이
        self.L3 = L3  # link 3 길이

        # ── DH 파라미터 [a, alpha, d, offset] ──
        self.dh = [
            [0,   -np.pi/2, L1, 0],
            [L2,   0,       0,  0],
            [L3,   0,       0,  0],
        ]

        # ── PoE: Home configuration M ──
        # q = [0,0,0]일 때 EE 위치
        self.M = np.array([
            [1, 0, 0, L2 + L3],
            [0, 1, 0, 0],
            [0, 0, 1, L1],
            [0, 0, 0, 1],
        ], dtype=float)

        # ── Space frame 스크류 축 Slist (6 x n_dof) ──
        self.Slist = np.array([
            [0, 0, 1, 0, 0, 0],           # joint 1: z축 회전 at origin
            [0, 1, 0, -L1, 0, 0],         # joint 2: y축 회전 at (0,0,L1)
            [0, 1, 0, -L1, 0, L2],        # joint 3: y축 회전 at (L2,0,L1)
        ]).T  # 6 x 3

        # ── 링크 물성 (동역학용) ──
        self.mass = [2.0, 1.5, 1.0]
        self.com = [
            np.array([0, 0, L1/2]),
            np.array([L2/2, 0, 0]),
            np.array([L3/2, 0, 0]),
        ]
        self.inertia = [np.eye(3) * 0.01 for _ in range(3)]

        # ── 관절 제한 ──
        self.q_min = np.array([-np.pi, -np.pi/2, -np.pi])
        self.q_max = np.array([ np.pi,  np.pi/2,  np.pi])
        self.tau_max = np.array([50.0, 30.0, 10.0])

        self.gravity = np.array([0, 0, -9.81])
