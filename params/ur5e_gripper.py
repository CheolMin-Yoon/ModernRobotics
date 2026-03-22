# -*- coding: utf-8 -*-
"""UR5e + 3지 그리퍼 통합 파라미터

kinematics_pick_and_place 프로젝트용.
UR5e 기구학은 params.ur5e에서 가져오고,
그리퍼 관절/액추에이터 설정을 추가.
"""

import numpy as np
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from params.ur5e import *  # noqa: F401,F403 — UR5e 기구학/동역학 전부 re-export

# ============================================================
# 경로 (그리퍼 포함 scene)
# ============================================================
PROJECT_DIR = os.path.join(os.path.dirname(__file__), '..', 'kinematics_pick_and_place')
GRIPPER_SCENE_XML = os.path.join(PROJECT_DIR, 'mujoco_gripper', 'scene.xml')
GRIPPER_URDF = os.path.join(PROJECT_DIR, 'description', 'urdf', 'ur5e_gripper.urdf')

# ============================================================
# UR5e 관절/액추에이터 이름
# ============================================================
UR5E_JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

UR5E_ACT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow",
    "wrist_1", "wrist_2", "wrist_3",
]

# ============================================================
# 3지 그리퍼 (각 손가락 4관절 × 3 = 12관절)
# ============================================================
GRIPPER_ACT_NAMES = [
    "center_palm_act", "center_upper_finger_act",
    "center_lower_finger_act", "center_fingertip_act",
    "left_palm_act", "left_upper_finger_act",
    "left_lower_finger_act", "left_fingertip_act",
    "right_palm_act", "right_upper_finger_act",
    "right_lower_finger_act", "right_fingertip_act",
]

FINGER_NAMES = ["center", "left", "right"]
JOINTS_PER_FINGER = 4
N_GRIPPER_JOINTS = len(FINGER_NAMES) * JOINTS_PER_FINGER  # 12

# EE / 물체 body 이름
EE_BODY = "gripper_palm"
OBJ_BODY = "target_object"

# ============================================================
# 그리퍼 자세 프리셋
# ============================================================
GRIPPER_OPEN = np.zeros(N_GRIPPER_JOINTS)
GRIPPER_CLOSE = np.array([
    0, 0.25, 0.25, 0.15,   # center
    0, 0.25, 0.25, 0.15,   # left
    0, 0.25, 0.25, 0.15,   # right
])

# ============================================================
# IK / 시뮬레이션 파라미터
# ============================================================
IK_MAX_ITER = 200
IK_TOL = 1e-3
IK_DAMPING = 1e-2

SIM_DT = 0.002
RENDER_HZ = 30
TRAJ_VEL = np.radians(30)
