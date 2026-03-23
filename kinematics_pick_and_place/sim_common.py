# -*- coding: utf-8 -*-
"""
Pick & Place 공통 유틸리티
- MujocoSim: MuJoCo 물리 시뮬 래퍼 (IK 없음)
- interp_joint_traj: 관절 공간 등속 보간
- ContactPlotter: 실시간 손가락별 접촉력 그래프
"""

import numpy as np
import mujoco
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from config import (
    SCENE_XML, UR5E_JOINT_NAMES, UR5E_ACT_NAMES,
    GRIPPER_ACT_NAMES, EE_BODY, OBJ_BODY,
    SIM_DT, TRAJ_VEL,
)


# ═══════════════════════════════════════════════════════
#  MuJoCo 시뮬 환경 (물리 + 렌더링만 담당)
# ═══════════════════════════════════════════════════════

class MujocoSim:
    """MuJoCo scene 래퍼 (IK 없음, 물리 시뮬 전용)"""

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(SCENE_XML)
        self.data = mujoco.MjData(self.model)

        self.arm_jnt_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
                            for n in UR5E_JOINT_NAMES]
        self.arm_qpos_ids = np.array([self.model.jnt_qposadr[j] for j in self.arm_jnt_ids])
        self.arm_dof_ids = np.array([self.model.jnt_dofadr[j] for j in self.arm_jnt_ids])
        self.arm_act_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                            for n in UR5E_ACT_NAMES]
        self.grip_act_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                             for n in GRIPPER_ACT_NAMES]
        self.ee_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY)
        self.obj_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, OBJ_BODY)
        self.key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_id)
        mujoco.mj_forward(self.model, self.data)

    def get_arm_q(self):
        return self.data.qpos[self.arm_qpos_ids].copy()

    def set_arm_q(self, q):
        self.data.qpos[self.arm_qpos_ids] = q

    def forward(self, q=None):
        if q is not None:
            self.set_arm_q(q)
        mujoco.mj_forward(self.model, self.data)

    def get_ee_pos(self):
        return self.data.xpos[self.ee_body_id].copy()

    def get_ee_rot(self):
        return self.data.xmat[self.ee_body_id].reshape(3, 3).copy()

    def get_obj_pos(self):
        return self.data.xpos[self.obj_body_id].copy()

    def get_arm_jacobian(self):
        """EE body의 (6×nv) 자코비안 → arm dof만 추출 → (6×6)"""
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.ee_body_id)
        J = np.vstack([jacp[:, self.arm_dof_ids],
                       jacr[:, self.arm_dof_ids]])
        return J  # (6×6)


# ═══════════════════════════════════════════════════════
#  궤적 생성: 관절 공간 등속 보간
# ═══════════════════════════════════════════════════════

def interp_joint_traj(waypoints, vel=TRAJ_VEL, hz=int(1/SIM_DT)):
    """waypoints: list of (n,) 관절각 → (N, n) 등속 보간 궤적"""
    waypoints = np.array(waypoints)
    segments = []
    for i in range(len(waypoints) - 1):
        q0, q1 = waypoints[i], waypoints[i+1]
        dist = np.max(np.abs(q1 - q0))
        duration = max(dist / vel, 0.5)
        n_steps = max(int(duration * hz), 2)
        seg = np.linspace(q0, q1, n_steps)
        segments.append(seg[:-1])
    segments.append(waypoints[-1:])
    return np.vstack(segments)


# ═══════════════════════════════════════════════════════
#  실시간 접촉력 그래프
# ═══════════════════════════════════════════════════════

FINGER_LABELS = ["Center", "Left", "Right"]
FINGER_COLORS = ["#e74c3c", "#2ecc71", "#3498db"]
FINGERTIP_GEOM_NAMES = [
    "center_fingertip_geom", "left_fingertip_geom", "right_fingertip_geom",
]
PLOT_WINDOW = 500


class ContactPlotter:
    """실시간 손가락별 접촉력 그래프"""

    def __init__(self, model, data, title='Fingertip Contact Force'):
        self.model = model
        self.data = data
        self.geom_to_finger = {}
        for i, name in enumerate(FINGERTIP_GEOM_NAMES):
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            self.geom_to_finger[gid] = i

        self.times = []
        self.finger_forces = [[] for _ in range(3)]
        self.total_force = []

        plt.ion()
        self.fig = plt.figure(figsize=(8, 5))
        gs = GridSpec(2, 1, height_ratios=[2, 1], hspace=0.3)

        self.ax_top = self.fig.add_subplot(gs[0])
        self.ax_top.set_title(title, fontsize=11)
        self.ax_top.set_ylabel('Force (N)')
        self.lines = []
        for i in range(3):
            line, = self.ax_top.plot([], [], color=FINGER_COLORS[i],
                                     label=FINGER_LABELS[i], linewidth=1.5)
            self.lines.append(line)
        self.ax_top.legend(loc='upper left', fontsize=9)
        self.ax_top.set_xlim(0, PLOT_WINDOW)
        self.ax_top.set_ylim(0, 5)
        self.ax_top.grid(True, alpha=0.3)

        self.ax_bot = self.fig.add_subplot(gs[1])
        self.ax_bot.set_title('Total Fingertip Force', fontsize=11)
        self.ax_bot.set_xlabel('Step')
        self.ax_bot.set_ylabel('Force (N)')
        self.line_total, = self.ax_bot.plot([], [], color='#e67e22', linewidth=1.5)
        self.ax_bot.set_xlim(0, PLOT_WINDOW)
        self.ax_bot.set_ylim(0, 10)
        self.ax_bot.grid(True, alpha=0.3)

        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.sample_count = 0

    def update(self):
        self.sample_count += 1
        finger_f = [0.0, 0.0, 0.0]
        for c_idx in range(self.data.ncon):
            contact = self.data.contact[c_idx]
            finger_idx = self.geom_to_finger.get(contact.geom1,
                         self.geom_to_finger.get(contact.geom2))
            if finger_idx is None:
                continue
            f_local = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, c_idx, f_local)
            finger_f[finger_idx] += np.linalg.norm(f_local[:3])

        for i in range(3):
            self.finger_forces[i].append(finger_f[i])
        self.total_force.append(sum(finger_f))
        self.times.append(self.sample_count)

        n = len(self.times)
        start = max(0, n - PLOT_WINDOW)
        t_slice = self.times[start:]

        y_max = 0.1
        for i in range(3):
            y = self.finger_forces[i][start:]
            self.lines[i].set_data(t_slice, y)
            if y:
                y_max = max(y_max, max(y) * 1.2)
        self.ax_top.set_xlim(t_slice[0], t_slice[-1])
        self.ax_top.set_ylim(0, max(y_max, 0.5))

        y_total = self.total_force[start:]
        self.line_total.set_data(t_slice, y_total)
        self.ax_bot.set_xlim(t_slice[0], t_slice[-1])
        y_max_t = max(max(y_total) * 1.2, 0.5) if y_total else 1.0
        self.ax_bot.set_ylim(0, y_max_t)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
