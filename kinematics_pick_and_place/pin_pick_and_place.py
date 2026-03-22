"""
Pinocchio IK + MuJoCo 시뮬 기반 Pick & Place
- FK/IK/Jacobian: Pinocchio (URDF)
- 물리 시뮬 + 렌더링: MuJoCo
- 궤적: 관절 공간 직선 보간 (등속)

실행:
  python kinematics_pick_and_place/pin_pick_and_place.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pinocchio as pin
import mujoco
import mujoco.viewer
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from config import *

np.set_printoptions(precision=4, suppress=True)

# ── Pinocchio URDF 경로 ──
UR5E_URDF = os.path.join(ROOT_DIR,
    'urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/'
    'universal_robots/ur_description/urdf/ur5e.urdf')


# ═══════════════════════════════════════════════════════
#  Pinocchio IK 헬퍼
# ═══════════════════════════════════════════════════════

class PinocchioIK:
    """Pinocchio 기반 FK/IK (UR5e URDF)"""

    def __init__(self, urdf_path, ee_frame='tool0'):
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.ee_fid = self.model.getFrameId(ee_frame)
        print(f"[Pinocchio] nq={self.model.nq}, nv={self.model.nv}, "
              f"ee_frame='{ee_frame}' (id={self.ee_fid})")

    def fk(self, q):
        """FK → (pos, rot)"""
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        oMf = self.data.oMf[self.ee_fid]
        return oMf.translation.copy(), oMf.rotation.copy()

    def jacobian(self, q):
        """Frame Jacobian (6×nv) in LOCAL_WORLD_ALIGNED"""
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return pin.getFrameJacobian(
            self.model, self.data, self.ee_fid,
            pin.LOCAL_WORLD_ALIGNED
        ).copy()

    def solve_ik(self, p_target, R_target, q_init,
                 max_iter=IK_MAX_ITER, eps=IK_TOL, dt=0.5, damp=IK_DAMPING):
        """Pinocchio 기반 damped least squares IK"""
        oMdes = pin.SE3(R_target, p_target)
        q = q_init.copy()

        for i in range(max_iter):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            oMcur = self.data.oMf[self.ee_fid]

            # SE(3) 오차
            iMd = oMcur.actInv(oMdes)
            err = pin.log(iMd).vector

            if np.linalg.norm(err) < eps:
                return q.copy(), True

            # Frame Jacobian (LOCAL)
            pin.computeJointJacobians(self.model, self.data, q)
            J = pin.getFrameJacobian(
                self.model, self.data, self.ee_fid, pin.LOCAL
            )

            # Jlog6 보정
            J = -pin.Jlog6(iMd.inverse()) @ J

            # damped least squares
            v = -J.T @ np.linalg.solve(
                J @ J.T + damp * np.eye(6), err)
            q = pin.integrate(self.model, q, v * dt)

        return q.copy(), np.linalg.norm(err) < eps


# ═══════════════════════════════════════════════════════
#  MuJoCo 시뮬 환경 (물리 + 렌더링만 담당)
# ═══════════════════════════════════════════════════════

class MujocoSim:
    """MuJoCo 물리 시뮬 + 렌더링 (IK는 Pinocchio가 담당)"""

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(SCENE_XML)
        self.data = mujoco.MjData(self.model)

        self.arm_jnt_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
                            for n in UR5E_JOINT_NAMES]
        self.arm_qpos_ids = np.array([self.model.jnt_qposadr[j] for j in self.arm_jnt_ids])
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

    def get_ee_pos(self):
        return self.data.xpos[self.ee_body_id].copy()

    def get_ee_rot(self):
        return self.data.xmat[self.ee_body_id].reshape(3, 3).copy()

    def get_obj_pos(self):
        return self.data.xpos[self.obj_body_id].copy()


# ═══════════════════════════════════════════════════════
#  궤적 생성
# ═══════════════════════════════════════════════════════

def interp_joint_traj(waypoints, vel=TRAJ_VEL, hz=int(1/SIM_DT)):
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
#  접촉력 그래프
# ═══════════════════════════════════════════════════════

FINGER_LABELS = ["Center", "Left", "Right"]
FINGER_COLORS = ["#e74c3c", "#2ecc71", "#3498db"]
FINGERTIP_GEOM_NAMES = ["center_fingertip_geom", "left_fingertip_geom", "right_fingertip_geom"]
PLOT_WINDOW = 500

class ContactPlotter:
    def __init__(self, model, data):
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
        self.ax_top.set_title('Fingertip Contact Force (Pinocchio IK)', fontsize=11)
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


# ═══════════════════════════════════════════════════════
#  메인
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    # Pinocchio IK 엔진
    pik = PinocchioIK(UR5E_URDF, ee_frame='tool0')

    # MuJoCo 시뮬 환경
    sim = MujocoSim()
    sim.reset()

    q_home = sim.get_arm_q()
    print(f"[Home] q = {np.rad2deg(q_home).round(1)}")

    # Pinocchio FK로 home EE 확인
    p_home_pin, R_home_pin = pik.fk(q_home)
    print(f"[Pinocchio FK] EE pos = {p_home_pin}")
    print(f"[MuJoCo FK]    EE pos = {sim.get_ee_pos()}")

    # MuJoCo에서 물체 위치 읽기
    obj_pos_mj = sim.get_obj_pos()
    print(f"[Obj pos (MuJoCo)] = {obj_pos_mj}")

    # ── MuJoCo ↔ Pinocchio 좌표 변환 ──
    # MuJoCo base에 quat="0 0 0 -1" (z축 180° 회전) 적용됨
    # MuJoCo → Pinocchio: x→-x, y→-y, z→z
    R_mj2pin = np.diag([-1.0, -1.0, 1.0])

    def mj2pin_pos(p):
        return R_mj2pin @ p

    def pin2mj_pos(p):
        return R_mj2pin @ p  # 자기 역행렬

    # 물체 위치를 Pinocchio 좌표로 변환
    obj_pos = mj2pin_pos(obj_pos_mj)

    # ── 그래스핑 자세 계산 ──
    R_grasp = R_home_pin.copy()
    home_z = p_home_pin[2]
    obj_top_z = obj_pos[2] + 0.05  # 박스 half-height

    # 웨이포인트 (Pinocchio 좌표계)
    p_approach = obj_pos.copy();    p_approach[2] = home_z
    p_pre_grasp = obj_pos.copy();   p_pre_grasp[2] = obj_top_z + 0.3
    p_grasp = obj_pos.copy();       p_grasp[2] = obj_top_z + 0.1
    p_lift = obj_pos.copy();        p_lift[2] = home_z
    p_place_above = obj_pos.copy(); p_place_above[0] += 0.25; p_place_above[1] -= 0.25; p_place_above[2] = home_z
    p_place = p_place_above.copy(); p_place[2] = obj_top_z + 0.1
    p_retreat = p_place.copy();     p_retreat[2] = home_z

    targets = [
        ("approach",    p_approach,    R_grasp),
        ("pre-grasp",   p_pre_grasp,   R_grasp),
        ("grasp",       p_grasp,       R_grasp),
        ("lift",        p_lift,        R_grasp),
        ("place_above", p_place_above, R_grasp),
        ("place",       p_place,       R_grasp),
        ("retreat",     p_retreat,     R_grasp),
    ]

    for name, p, _ in targets:
        print(f"[목표 (pin)] {name:12s}: {p}")
        print(f"[목표 (mj)]  {name:12s}: {pin2mj_pos(p)}")

    # ── Pinocchio IK 풀기 ──
    print("\n--- Pinocchio IK 풀기 ---")
    q_solutions = [q_home]
    q_prev = q_home.copy()

    for name, p_tgt, R_tgt in targets:
        q_sol, converged = pik.solve_ik(p_tgt, R_tgt, q_prev)
        p_fk, _ = pik.fk(q_sol)
        err = np.linalg.norm(p_tgt - p_fk)
        status = "OK" if converged else "FAIL"
        print(f"  [{name:12s}] {status}  err={err*1000:.2f}mm  q={np.rad2deg(q_sol).round(1)}")
        q_solutions.append(q_sol)
        q_prev = q_sol.copy()

    q_solutions.append(q_home)

    # ── 궤적 생성 ──
    traj = interp_joint_traj(q_solutions)
    print(f"\n[궤적] {len(traj)} steps ({len(traj)*SIM_DT:.1f}s)")

    seg_lengths = []
    for i in range(len(q_solutions) - 1):
        dist = np.max(np.abs(q_solutions[i+1] - q_solutions[i]))
        dur = max(dist / TRAJ_VEL, 0.5)
        seg_lengths.append(max(int(dur / SIM_DT), 2))

    cumulative = np.cumsum(seg_lengths)
    grasp_idx = cumulative[2]
    place_idx = cumulative[5]
    print(f"[그리퍼] close at step {grasp_idx}, open at step {place_idx}")

    # ═══════════════════════════════════════════════════════
    #  MuJoCo 시뮬레이션 실행
    # ═══════════════════════════════════════════════════════

    sim.reset()
    render_every = max(1, int(1 / (RENDER_HZ * SIM_DT)))

    waypoint_positions = [pin2mj_pos(p_tgt) for (_, p_tgt, _) in targets]
    waypoint_colors = [
        [0, 1, 0, 0.6], [0, 0.7, 1, 0.6], [1, 0, 0, 0.8],
        [1, 1, 0, 0.6], [0, 0.7, 1, 0.6], [1, 0.5, 0, 0.8],
        [0.5, 0, 1, 0.6],
    ]

    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

        contact_plotter = ContactPlotter(sim.model, sim.data)
        plot_every = render_every * 5

        # 웨이포인트 구체
        for i, (pos, rgba) in enumerate(zip(waypoint_positions, waypoint_colors)):
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[i],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.015, 0, 0],
                pos=pos.astype(np.float64),
                mat=np.eye(3).flatten().astype(np.float64),
                rgba=np.array(rgba, dtype=np.float32),
            )
        viewer.user_scn.ngeom = len(waypoint_positions)
        viewer.sync()

        tick = 0
        gripper_closed = False
        waypoint_idx = 0

        while viewer.is_running() and tick < len(traj):
            step_start = time.time()
            q_target = traj[tick]

            if tick >= grasp_idx and tick < place_idx:
                if not gripper_closed:
                    gripper_closed = True
                    print(f"[t={tick*SIM_DT:.2f}s] 그리퍼 닫기")
                grip_cmd = GRIPPER_CLOSE
            else:
                if gripper_closed and tick >= place_idx:
                    gripper_closed = False
                    print(f"[t={tick*SIM_DT:.2f}s] 그리퍼 열기")
                grip_cmd = GRIPPER_OPEN

            sim.data.ctrl[sim.arm_act_ids] = q_target
            sim.data.ctrl[sim.grip_act_ids] = grip_cmd
            mujoco.mj_step(sim.model, sim.data)

            if tick % render_every == 0:
                viewer.sync()
            if tick % plot_every == 0:
                contact_plotter.update()

            tick += 1

            elapsed = time.time() - step_start
            sleep_time = SIM_DT - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            if waypoint_idx < len(cumulative) and tick == cumulative[waypoint_idx]:
                delay = WAYPOINT_DELAYS[waypoint_idx] if waypoint_idx < len(WAYPOINT_DELAYS) else 0.5
                print(f"[t={tick*SIM_DT:.2f}s] 웨이포인트 {waypoint_idx} 도달, {delay}초 hold")
                hold_remaining = delay
                while viewer.is_running() and hold_remaining > 0:
                    sim.data.ctrl[sim.arm_act_ids] = q_target
                    sim.data.ctrl[sim.grip_act_ids] = grip_cmd
                    mujoco.mj_step(sim.model, sim.data)
                    viewer.sync()
                    time.sleep(SIM_DT)
                    hold_remaining -= SIM_DT
                waypoint_idx += 1

        print("\n[완료] Pinocchio IK Pick & Place 시뮬레이션 종료")
        remaining = 3.0
        while viewer.is_running() and remaining > 0:
            mujoco.mj_step(sim.model, sim.data)
            viewer.sync()
            time.sleep(SIM_DT)
            remaining -= SIM_DT

    print("[종료]")
