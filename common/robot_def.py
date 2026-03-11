"""
로봇 정의 — 모든 챕터에서 공유

3DOF RRR 매니퓰레이터를 기본 실습 로봇으로 사용.
PoE (Product of Exponentials) 파라미터와 DH 파라미터 모두 정의.
URDF: common/3_dof_manipulator.urdf
"""

import numpy as np
import os


class ThreeLinkRobot:
    """3DOF RRR Planar-ish Manipulator"""

    def __init__(self):
        self.n_dof = 3
        
        # URDF 기반 링크 길이
        self.L1 = 0.075   # base_link to link1 (joint1 origin z)
        self.L2 = 0.125   # link1 to link2 (joint2 origin z)
        self.L3 = 0.125   # link2 to end_effector (joint3 origin z)
        
        # URDF 파일 경로
        self.urdf_path = os.path.join(
            os.path.dirname(__file__), 
            '3_dof_manipulator.urdf'
        )

        # ── DH 파라미터 [a, alpha, d, offset] ──
        # Joint1: z축 회전, d=0.075
        # Joint2: y축 회전, d=0.125 (z방향 오프셋)
        # Joint3: y축 회전, d=0.125 (z방향 오프셋)
        self.dh = [
            [0,   -np.pi/2, self.L1, 0],  # joint1: z-axis rotation
            [0,    0,       self.L2, 0],  # joint2: y-axis rotation
            [0,    0,       self.L3, 0],  # joint3: y-axis rotation
        ]

        # ── PoE: Home configuration M ──
        # q = [0,0,0]일 때 EE 위치 (z축 누적)
        total_height = self.L1 + self.L2 + self.L3
        self.M = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, total_height],
            [0, 0, 0, 1],
        ], dtype=float)

        # ── Space frame 스크류 축 Slist (6 x n_dof) ──
        self.Slist = np.array([
            [0, 0, 1, 0, 0, 0],                    # joint1: z축 회전 at (0,0,0)
            [0, 1, 0, -self.L1, 0, 0],             # joint2: y축 회전 at (0,0,L1)
            [0, 1, 0, -(self.L1+self.L2), 0, 0],   # joint3: y축 회전 at (0,0,L1+L2)
        ]).T  # 6 x 3

        # ── 링크 물성 (동역학용) ──
        # URDF의 inertial 값 반영
        self.mass = [0.5, 0.2, 0.2, 0.1]  # base_link, link1, link2, end_effector
        self.com = [
            np.array([0, 0, 0]),           # base_link
            np.array([0, 0, 0.1]),         # link1 (box 0.2 길이의 중심)
            np.array([0, 0, 0.1]),         # link2 (box 0.2 길이의 중심)
            np.array([0, 0, 0.025]),       # end_effector (box 0.05 길이의 중심)
        ]
        self.inertia = [
            np.diag([0.0004, 0.0004, 0.0004]),  # base_link
            np.diag([0.0002, 0.0002, 0.0002]),  # link1
            np.diag([0.0002, 0.0002, 0.0002]),  # link2
            np.diag([0.00005, 0.00005, 0.00005]),  # end_effector
        ]

        # ── 관절 제한 ──
        self.q_min = np.array([-np.pi, -np.pi/2, -np.pi])
        self.q_max = np.array([ np.pi,  np.pi/2,  np.pi])
        self.tau_max = np.array([50.0, 30.0, 10.0])

        self.gravity = np.array([0, 0, -9.81])
