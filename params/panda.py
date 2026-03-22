# -*- coding: utf-8 -*-
"""Franka Emika Panda 기구학 파라미터 (단위: m)

7DOF 매니퓰레이터. MuJoCo menagerie + URDF 기준.
TODO: 스크류 축, M 행렬 등 구현 예정
"""

import numpy as np
import os

# ============================================================
# 경로
# ============================================================
URDF_PATH = os.path.join(os.path.dirname(__file__), '..',
    'urdf_files_dataset/urdf_files/robotics-toolbox/xacro_generated/'
    'franka_description/robots/panda.urdf')

MUJOCO_SCENE = os.path.join(os.path.dirname(__file__), '..',
    'mujoco_menagerie/franka_emika_panda/panda.xml')

# ============================================================
# DH / 링크 파라미터
# ============================================================
# Panda 링크 길이 (m) - Franka 공식 문서 기준
d1 = 0.333
d3 = 0.316
d5 = 0.384
df = 0.107   # flange
a4 = 0.0825
a5 = -0.0825
a7 = 0.088

# TODO: 스크류 축, M 행렬, 동역학 파라미터 추가 예정
