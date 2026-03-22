#!/usr/bin/env python3
"""
MuJoCo 3-Finger Gripper 시뮬레이션 제어 + 실시간 센서 플롯

사용법:
  python gripper_sim_control.py                # grasp 데모 + 실시간 그래프
  python gripper_sim_control.py --mode free    # viewer 자유 조작 + 실시간 그래프
  python gripper_sim_control.py --no-plot      # 그래프 없이 실행

조인트 매핑 (SDK motor_id → MuJoCo joint):
  Finger1 (center): ID 1→center_palm, 2→center_upper_finger, 3→center_lower_finger
  Finger2 (left):   ID 4→left_palm,   5→left_upper_finger,   6→left_lower_finger,   7→left_fingertip
  Finger3 (right):  ID 8→right_palm,  9→right_upper_finger,  10→right_lower_finger,  11→right_fingertip
"""
import argparse
import time
import threading
from collections import deque
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCENE_XML  = os.path.join(SCRIPT_DIR, "scene.xml")

UR5E_ACTUATORS = [
    "shoulder_pan", "shoulder_lift", "elbow",
    "wrist_1", "wrist_2", "wrist_3",
]

GRIPPER_ACTUATORS = [
    "center_palm_act", "center_upper_finger_act",
    "center_lower_finger_act", "center_fingertip_act",
    "left_palm_act", "left_upper_finger_act",
    "left_lower_finger_act", "left_fingertip_act",
    "right_palm_act", "right_upper_finger_act",
    "right_lower_finger_act", "right_fingertip_act",
]


# ── 실시간 플롯 ──────────────────────────────────────────────

class RealtimePlotter:
    """별도 스레드에서 matplotlib 실시간 그래프 표시"""

    def __init__(self, max_points=500):
        self.max_points = max_points
        self.lock = threading.Lock()

        # 데이터 버퍼
        self.t_buf = deque(maxlen=max_points)
        self.touch_center = deque(maxlen=max_points)
        self.touch_left   = deque(maxlen=max_points)
        self.touch_right  = deque(maxlen=max_points)
        self.ee_fx = deque(maxlen=max_points)
        self.ee_fy = deque(maxlen=max_points)
        self.ee_fz = deque(maxlen=max_points)
        self.ur5e_q = [deque(maxlen=max_points) for _ in range(6)]

        self._running = True
        self._t0 = time.time()

    def push(self, touch: dict, wrench: dict, ur5e: dict):
        """시뮬 루프에서 호출 — 센서 데이터 push"""
        with self.lock:
            t = time.time() - self._t0
            self.t_buf.append(t)
            self.touch_center.append(touch.get("center_touch_sensor", 0))
            self.touch_left.append(touch.get("left_touch_sensor", 0))
            self.touch_right.append(touch.get("right_touch_sensor", 0))
            f = wrench.get("force", np.zeros(3))
            self.ee_fx.append(f[0])
            self.ee_fy.append(f[1])
            self.ee_fz.append(f[2])
            q = ur5e.get("q", np.zeros(6))
            for i in range(6):
                self.ur5e_q[i].append(np.degrees(q[i]))

    def run_plot(self):
        """메인 스레드에서 호출 — matplotlib 이벤트 루프 실행"""
        self.fig, self.axes = plt.subplots(3, 1, figsize=(10, 8))
        self.fig.suptitle("Gripper Sensor Monitor", fontsize=13)
        self.fig.tight_layout(rect=[0, 0, 1, 0.95], h_pad=3)

        # subplot 0: Touch forces
        ax0 = self.axes[0]
        ax0.set_title("Fingertip Touch Force (N)")
        ax0.set_ylabel("Force (N)")
        self.line_tc, = ax0.plot([], [], 'r-', label='center', linewidth=1.2)
        self.line_tl, = ax0.plot([], [], 'g-', label='left', linewidth=1.2)
        self.line_tr, = ax0.plot([], [], 'b-', label='right', linewidth=1.2)
        ax0.legend(loc='upper right', fontsize=8)
        ax0.set_xlim(0, 10)
        ax0.set_ylim(-0.5, 20)
        ax0.grid(True, alpha=0.3)

        # subplot 1: EE force
        ax1 = self.axes[1]
        ax1.set_title("EE Force (N)")
        ax1.set_ylabel("Force (N)")
        self.line_fx, = ax1.plot([], [], 'r-', label='Fx', linewidth=1.2)
        self.line_fy, = ax1.plot([], [], 'g-', label='Fy', linewidth=1.2)
        self.line_fz, = ax1.plot([], [], 'b-', label='Fz', linewidth=1.2)
        ax1.legend(loc='upper right', fontsize=8)
        ax1.set_xlim(0, 10)
        ax1.set_ylim(-50, 50)
        ax1.grid(True, alpha=0.3)

        # subplot 2: UR5e joint positions
        ax2 = self.axes[2]
        ax2.set_title("UR5e Joint Positions (deg)")
        ax2.set_ylabel("Angle (deg)")
        ax2.set_xlabel("Time (s)")
        colors = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#42d4f4']
        names = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6']
        self.line_jq = []
        for i in range(6):
            ln, = ax2.plot([], [], color=colors[i], label=names[i], linewidth=1.0)
            self.line_jq.append(ln)
        ax2.legend(loc='upper right', fontsize=7, ncol=3)
        ax2.set_xlim(0, 10)
        ax2.set_ylim(-200, 200)
        ax2.grid(True, alpha=0.3)

        self.anim = FuncAnimation(self.fig, self._update, interval=50, blit=False, cache_frame_data=False)
        plt.show()

    def _update(self, frame):
        with self.lock:
            if len(self.t_buf) < 2:
                return
            t = list(self.t_buf)
            t_max = t[-1]
            t_min = max(0, t_max - 10)

            # Touch
            self.line_tc.set_data(t, list(self.touch_center))
            self.line_tl.set_data(t, list(self.touch_left))
            self.line_tr.set_data(t, list(self.touch_right))
            self.axes[0].set_xlim(t_min, t_max + 0.5)
            # auto y
            all_touch = list(self.touch_center) + list(self.touch_left) + list(self.touch_right)
            if all_touch:
                ymax = max(max(all_touch), 1.0) * 1.2
                self.axes[0].set_ylim(-0.5, ymax)

            # EE force
            self.line_fx.set_data(t, list(self.ee_fx))
            self.line_fy.set_data(t, list(self.ee_fy))
            self.line_fz.set_data(t, list(self.ee_fz))
            self.axes[1].set_xlim(t_min, t_max + 0.5)

            # UR5e joints
            for i in range(6):
                self.line_jq[i].set_data(t, list(self.ur5e_q[i]))
            self.axes[2].set_xlim(t_min, t_max + 0.5)

    def stop(self):
        self._running = False
        plt.close('all')


# ── 컨트롤러 ─────────────────────────────────────────────────

class GripperSimController:
    """MuJoCo 시뮬레이션에서 UR5e + 3-finger gripper 제어"""

    def __init__(self, scene_xml=SCENE_XML):
        self.model = mujoco.MjModel.from_xml_path(scene_xml)
        self.data  = mujoco.MjData(self.model)

        self.ur5e_act_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in UR5E_ACTUATORS
        ]
        self.gripper_act_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in GRIPPER_ACTUATORS
        ]

        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        mujoco.mj_forward(self.model, self.data)

    def set_ur5e_joints(self, q: np.ndarray):
        for i, aid in enumerate(self.ur5e_act_ids):
            self.data.ctrl[aid] = q[i]

    def set_gripper_joints(self, q: np.ndarray):
        for i, aid in enumerate(self.gripper_act_ids):
            self.data.ctrl[aid] = q[i]

    def set_gripper_from_sdk(self, motor_cmds: dict):
        gripper_q = np.zeros(12)
        gripper_q[0] = motor_cmds.get(1, 0.0)
        gripper_q[1] = motor_cmds.get(2, 0.0)
        gripper_q[2] = motor_cmds.get(3, 0.0)
        gripper_q[3] = motor_cmds.get(3, 0.0)  # coupled
        gripper_q[4] = motor_cmds.get(4, 0.0)
        gripper_q[5] = motor_cmds.get(5, 0.0)
        gripper_q[6] = motor_cmds.get(6, 0.0)
        gripper_q[7] = motor_cmds.get(7, 0.0)
        gripper_q[8]  = motor_cmds.get(8, 0.0)
        gripper_q[9]  = motor_cmds.get(9, 0.0)
        gripper_q[10] = motor_cmds.get(10, 0.0)
        gripper_q[11] = motor_cmds.get(11, 0.0)
        self.set_gripper_joints(gripper_q)

    def get_gripper_state(self) -> dict:
        state = {}
        for name in GRIPPER_ACTUATORS:
            jname = name.replace("_act", "_joint")
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid >= 0:
                state[jname] = self.data.qpos[self.model.jnt_qposadr[jid]]
        return state

    def step(self):
        mujoco.mj_step(self.model, self.data)

    def get_touch_forces(self) -> dict:
        result = {}
        for name in ["center_touch_sensor", "left_touch_sensor", "right_touch_sensor"]:
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            if sid >= 0:
                result[name] = self.data.sensordata[self.model.sensor_adr[sid]]
        return result

    def get_ee_wrench(self) -> dict:
        f_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "ee_force")
        t_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "ee_torque")
        return {
            'force':  self.data.sensordata[self.model.sensor_adr[f_id]:self.model.sensor_adr[f_id]+3].copy(),
            'torque': self.data.sensordata[self.model.sensor_adr[t_id]:self.model.sensor_adr[t_id]+3].copy(),
        }

    def get_ur5e_state(self) -> dict:
        q, dq = np.zeros(6), np.zeros(6)
        for i, prefix in enumerate(["shoulder_pan", "shoulder_lift", "elbow",
                                     "wrist_1", "wrist_2", "wrist_3"]):
            jp_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, f"jp_{prefix}")
            jv_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, f"jv_{prefix}")
            q[i]  = self.data.sensordata[self.model.sensor_adr[jp_id]]
            dq[i] = self.data.sensordata[self.model.sensor_adr[jv_id]]
        return {'q': q, 'dq': dq}

    def grasp_motion(self, amplitude=0.8, duration=3.0, dt=0.002):
        steps = int(duration / dt)
        trajectory = []
        for i in range(steps):
            t = i * dt
            angle = amplitude * min(1.0, t / (duration * 0.5))
            cmd = {
                1: 0.0, 2: angle, 3: angle,
                4: 0.0, 5: angle, 6: angle, 7: angle,
                8: 0.0, 9: angle, 10: angle, 11: angle,
            }
            trajectory.append(cmd)
        return trajectory


# ── 실행 모드 ────────────────────────────────────────────────

def sim_loop(ctrl: GripperSimController, plotter: RealtimePlotter = None, mode="grasp"):
    """시뮬 루프 (별도 스레드에서 실행)"""
    ur5e_home = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
    ctrl.set_ur5e_joints(ur5e_home)

    trajectory = ctrl.grasp_motion(amplitude=0.8, duration=3.0)
    plot_interval = 10  # 10 step마다 플롯 데이터 push (= 50Hz)

    with mujoco.viewer.launch_passive(ctrl.model, ctrl.data) as viewer:
        step = 0
        direction = 1
        traj_idx = 0

        while viewer.is_running():
            t0 = time.time()

            if mode == "grasp":
                if direction == 1:
                    ctrl.set_gripper_from_sdk(trajectory[min(traj_idx, len(trajectory)-1)])
                    traj_idx += 1
                    if traj_idx >= len(trajectory):
                        direction = -1
                        traj_idx = len(trajectory) - 1
                else:
                    ctrl.set_gripper_from_sdk(trajectory[max(traj_idx, 0)])
                    traj_idx -= 1
                    if traj_idx <= 0:
                        direction = 1
                        traj_idx = 0

            ctrl.step()
            viewer.sync()

            # 센서 데이터 → 플롯
            if plotter and step % plot_interval == 0:
                touch  = ctrl.get_touch_forces()
                wrench = ctrl.get_ee_wrench()
                ur5e   = ctrl.get_ur5e_state()
                plotter.push(touch, wrench, ur5e)

            step += 1
            elapsed = time.time() - t0
            sleep_time = ctrl.model.opt.timestep - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    if plotter:
        plotter.stop()


def main():
    parser = argparse.ArgumentParser(description="MuJoCo 3-Finger Gripper Control")
    parser.add_argument("--mode", choices=["grasp", "free"], default="grasp")
    parser.add_argument("--scene", default=SCENE_XML)
    parser.add_argument("--no-plot", action="store_true", help="실시간 그래프 비활성화")
    args = parser.parse_args()

    ctrl = GripperSimController(args.scene)

    if args.no_plot:
        sim_loop(ctrl, plotter=None, mode=args.mode)
    else:
        plotter = RealtimePlotter(max_points=500)

        # 시뮬 루프는 별도 스레드
        sim_thread = threading.Thread(
            target=sim_loop, args=(ctrl, plotter, args.mode), daemon=True
        )
        sim_thread.start()

        # matplotlib은 메인 스레드에서 실행 (필수)
        print("=== 실시간 센서 모니터 ===")
        print("MuJoCo viewer + matplotlib 그래프 동시 실행")
        print("viewer 닫으면 종료\n")
        plotter.run_plot()


if __name__ == "__main__":
    main()
