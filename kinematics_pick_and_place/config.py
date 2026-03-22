"""
Pick & Place 공통 설정
UR5e + 3-finger gripper (robot_hand/mujoco_gripper)
"""
import os
import numpy as np

# ── 경로 ──
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(PROJECT_DIR, '..')
SCENE_XML = os.path.join(PROJECT_DIR, 'mujoco_gripper', 'scene.xml')

# ── UR5e 관절 이름 (6DOF) ──
UR5E_JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

# ── UR5e 액추에이터 이름 ──
UR5E_ACT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow",
    "wrist_1", "wrist_2", "wrist_3",
]

# ── 그리퍼 액추에이터 (3-finger: 각 손가락 4관절 × 3) ──
GRIPPER_ACT_NAMES = [
    "center_palm_act", "center_upper_finger_act",
    "center_lower_finger_act", "center_fingertip_act",
    "left_palm_act", "left_upper_finger_act",
    "left_lower_finger_act", "left_fingertip_act",
    "right_palm_act", "right_upper_finger_act",
    "right_lower_finger_act", "right_fingertip_act",
]

# ── EE body 이름 ──
EE_BODY = "gripper_palm"

# ── 오브젝트 ──
OBJ_BODY = "target_object"

# ── IK 파라미터 ──
IK_MAX_ITER = 200
IK_TOL = 1e-3
IK_STEP_SIZE = 0.5
IK_DAMPING = 1e-2
IK_DQ_MAX = 5.0 * np.pi / 180.0  # 프레임당 최대 관절 변화

# ── 궤적 보간 ──
TRAJ_VEL = np.radians(30)  # 관절 공간 보간 속도 (rad/s)

# ── 그리퍼 자세 ──
GRIPPER_OPEN = np.zeros(12)
GRIPPER_CLOSE = np.array([
    0, 0.25, 0.25, 0.15,   # center
    0, 0.25, 0.25, 0.15,   # left
    0, 0.25, 0.25, 0.15,   # right
])

# ── 시뮬레이션 ──
SIM_DT = 0.002  # scene.xml 기본값
RENDER_HZ = 30

# ── 구간별 딜레이 (초) ──
# 각 웨이포인트 도달 후 hold 시간
# 순서: approach, pre-grasp, grasp, lift, place_above, place, retreat, home
WAYPOINT_DELAYS = [0.5, 0.5, 1.0, 0.5, 0.5, 1.0, 0.5, 0.5]
