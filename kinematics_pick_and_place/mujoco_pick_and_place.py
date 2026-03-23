"""
MuJoCo API 기반 Pick & Place
- FK: data.body(name).xpos / xmat
- 자코비안: mj_jacBody
- IK: damped least squares
- 궤적: 관절 공간 직선 보간 (등속)
- 시뮬: mj_step

실행:
  python kinematics_pick_and_place/mujoco_pick_and_place.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import mujoco
import mujoco.viewer
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import threading

from config import *
from sim_common import MujocoSim, interp_joint_traj
from grasp_analysis import GraspAnalyzer

np.set_printoptions(precision=4, suppress=True)


# ═══════════════════════════════════════════════════════
#  유틸리티
# ═══════════════════════════════════════════════════════

def rpy2r(rpy):
    """Roll-Pitch-Yaw (rad) → 3×3 회전행렬"""
    cr, sr = np.cos(rpy[0]), np.sin(rpy[0])
    cp, sp = np.cos(rpy[1]), np.sin(rpy[1])
    cy, sy = np.cos(rpy[2]), np.sin(rpy[2])
    return np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr],
    ])


def r2w(R):
    """회전행렬 → angular velocity vector (로그맵 근사)"""
    el = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])
    norm = np.linalg.norm(el)
    if norm > 1e-10:
        return np.arctan2(norm, np.trace(R)-1) / norm * el
    return np.zeros(3)


def trim_scale(dq, th):
    """dq 벡터의 최대 절대값을 th로 클리핑"""
    mx = np.abs(dq).max()
    if mx > th:
        dq = dq * th / mx
    return dq


# ═══════════════════════════════════════════════════════
#  MuJoCo 헬퍼
# ═══════════════════════════════════════════════════════

class MujocoEnv(MujocoSim):
    """MujocoSim + MuJoCo 자코비안 기반 IK"""

    # ── IK: damped least squares ──
    def solve_ik(self, p_target, R_target, q_init,
                 ik_p=True, ik_r=True, w_weight=0.3):
        """
        MuJoCo 자코비안 기반 수치 IK.
        p_target: (3,) 목표 위치
        R_target: (3,3) 목표 자세
        q_init: (6,) 초기 관절각
        """
        q = q_init.copy()
        self.forward(q)

        for _ in range(IK_MAX_ITER):
            p_curr = self.get_ee_pos()
            R_curr = self.get_ee_rot()

            # 오차 계산
            if ik_p and ik_r:
                p_err = p_target - p_curr
                R_err = np.linalg.solve(R_curr, R_target)
                w_err = R_curr @ r2w(R_err) * w_weight
                err = np.concatenate([p_err, w_err])
                J = self.get_arm_jacobian()
            elif ik_p:
                err = p_target - p_curr
                J = self.get_arm_jacobian()[:3]
            else:
                R_err = np.linalg.solve(R_curr, R_target)
                w_err = R_curr @ r2w(R_err)
                err = w_err
                J = self.get_arm_jacobian()[3:]

            if np.linalg.norm(err) < IK_TOL:
                return q, True

            # damped least squares
            dq = IK_STEP_SIZE * np.linalg.solve(
                J.T @ J + IK_DAMPING * np.eye(6), J.T @ err)
            dq = trim_scale(dq, IK_DQ_MAX)

            q = q + dq
            self.forward(q)

        return q, np.linalg.norm(err) < IK_TOL


# ═══════════════════════════════════════════════════════
#  접촉점 시각화 헬퍼
# ═══════════════════════════════════════════════════════

def get_contact_geoms(model, data, max_contacts=20):
    """
    접촉점 정보 수집 → user_scn에 그릴 geom 데이터 리스트 반환.
    각 접촉점마다 구체(위치) + 화살표(힘 방향) 2개.
    """
    geoms = []
    for c_idx in range(data.ncon):
        if len(geoms) >= max_contacts * 2:
            break
        contact = data.contact[c_idx]
        p = contact.pos.copy()
        R_frame = contact.frame.reshape(3, 3)

        # 접촉력 (로컬 → 글로벌)
        f_local = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, c_idx, f_local)
        f_global = R_frame @ f_local[:3]
        f_norm = np.linalg.norm(f_global)

        if f_norm < 1e-6:
            continue

        # 구체: 접촉점 위치
        geoms.append({
            'type': mujoco.mjtGeom.mjGEOM_SPHERE,
            'size': [0.008, 0, 0],
            'pos': p,
            'mat': np.eye(3).flatten(),
            'rgba': np.array([1, 0.2, 0.2, 0.9], dtype=np.float32),
        })

        # 화살표: 힘 방향
        f_uv = f_global / f_norm
        h_arrow = min(f_norm * 0.005, 0.15)  # 힘 크기에 비례, 최대 15cm
        # z축 → f_uv 방향 회전행렬
        z = np.array([0, 0, 1.0])
        v = np.cross(z, f_uv)
        s = np.linalg.norm(v)
        c = np.dot(z, f_uv)
        if s > 1e-9:
            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            R_arrow = np.eye(3) + vx + vx @ vx * (1 - c) / (s * s)
        else:
            R_arrow = np.eye(3) if c > 0 else np.diag([1, -1, -1.0])

        geoms.append({
            'type': mujoco.mjtGeom.mjGEOM_ARROW,
            'size': [0.005, 0.005, h_arrow],
            'pos': p,
            'mat': R_arrow.flatten(),
            'rgba': np.array([1, 0.5, 0, 0.8], dtype=np.float32),
        })

    return geoms


# ═══════════════════════════════════════════════════════
#  실시간 접촉력 그래프
# ═══════════════════════════════════════════════════════

from sim_common import (
    FINGER_LABELS, FINGER_COLORS, FINGERTIP_GEOM_NAMES, PLOT_WINDOW,
)


class ContactPlotter:
    """실시간 손가락별 접촉력 + 그래스프 분석 그래프"""

    def __init__(self, model, data):
        self.model = model
        self.data = data

        # fingertip geom ID → finger index 매핑
        self.geom_to_finger = {}
        for i, name in enumerate(FINGERTIP_GEOM_NAMES):
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            self.geom_to_finger[gid] = i
        self.fingertip_geom_ids = set(self.geom_to_finger.keys())

        # 그래스프 분석기
        self.grasp_analyzer = GraspAnalyzer(model, data, FINGERTIP_GEOM_NAMES, OBJ_BODY)

        # 데이터 버퍼
        self.times = []
        self.finger_forces = [[] for _ in range(3)]
        self.total_force = []
        self.grasp_quality = []
        self.grasp_rank = []
        self.closed_chain_dof = []

        # matplotlib 설정
        plt.ion()
        self.fig = plt.figure(figsize=(10, 8))
        gs = GridSpec(3, 1, height_ratios=[2, 1, 1.5], hspace=0.35)

        # 상단: 각 손가락 접촉력
        self.ax_touch = self.fig.add_subplot(gs[0])
        self.ax_touch.set_title('Fingertip Contact Force', fontsize=11)
        self.ax_touch.set_ylabel('Force (N)')
        self.lines_touch = []
        for i in range(3):
            line, = self.ax_touch.plot([], [], color=FINGER_COLORS[i],
                                       label=FINGER_LABELS[i], linewidth=1.5)
            self.lines_touch.append(line)
        self.ax_touch.legend(loc='upper left', fontsize=9)
        self.ax_touch.set_xlim(0, PLOT_WINDOW)
        self.ax_touch.set_ylim(0, 5)
        self.ax_touch.grid(True, alpha=0.3)

        # 중단: 총 접촉력
        self.ax_total = self.fig.add_subplot(gs[1])
        self.ax_total.set_title('Total Fingertip Force', fontsize=11)
        self.ax_total.set_ylabel('Force (N)')
        self.line_total, = self.ax_total.plot([], [], color='#e67e22', linewidth=1.5)
        self.ax_total.set_xlim(0, PLOT_WINDOW)
        self.ax_total.set_ylim(0, 10)
        self.ax_total.grid(True, alpha=0.3)

        # 하단: 그래스프 분석 (quality + rank + DOF)
        self.ax_grasp = self.fig.add_subplot(gs[2])
        self.ax_grasp.set_title('Grasp Analysis (ch05/ch07)', fontsize=11)
        self.ax_grasp.set_xlabel('Step')
        self.ax_grasp.set_ylabel('Quality / Rank')
        self.line_quality, = self.ax_grasp.plot([], [], color='#9b59b6',
                                                 label='Quality (isotropy)', linewidth=1.5)
        self.line_rank, = self.ax_grasp.plot([], [], color='#1abc9c',
                                              label='G rank / 6', linewidth=1.5, linestyle='--')
        self.line_dof, = self.ax_grasp.plot([], [], color='#e74c3c',
                                             label='Obj DOF / 6', linewidth=1.5, linestyle=':')
        self.ax_grasp.legend(loc='upper left', fontsize=8)
        self.ax_grasp.set_xlim(0, PLOT_WINDOW)
        self.ax_grasp.set_ylim(-0.05, 1.1)
        self.ax_grasp.axhline(y=1.0, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
        self.ax_grasp.grid(True, alpha=0.3)

        # Force closure 텍스트
        self.fc_text = self.ax_grasp.text(0.98, 0.95, '', transform=self.ax_grasp.transAxes,
                                           ha='right', va='top', fontsize=10,
                                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.sample_count = 0

    def update(self):
        """접촉 데이터 + 그래스프 분석 갱신"""
        self.sample_count += 1

        # 각 손가락별 접촉력 계산
        finger_f = [0.0, 0.0, 0.0]
        for c_idx in range(self.data.ncon):
            contact = self.data.contact[c_idx]
            finger_idx = None
            if contact.geom1 in self.geom_to_finger:
                finger_idx = self.geom_to_finger[contact.geom1]
            elif contact.geom2 in self.geom_to_finger:
                finger_idx = self.geom_to_finger[contact.geom2]
            if finger_idx is None:
                continue
            f_local = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, c_idx, f_local)
            finger_f[finger_idx] += np.linalg.norm(f_local[:3])

        for i in range(3):
            self.finger_forces[i].append(finger_f[i])
        self.total_force.append(sum(finger_f))

        # 그래스프 분석
        result = self.grasp_analyzer.analyze()
        self.grasp_quality.append(result['grasp_quality'])
        self.grasp_rank.append(result['G_rank'] / 6.0)
        self.closed_chain_dof.append(result['object_dof'] / 6.0)

        self.times.append(self.sample_count)

        # 윈도우 범위
        n = len(self.times)
        start = max(0, n - PLOT_WINDOW)
        t_slice = self.times[start:]

        # 상단 갱신
        y_max_touch = 0.1
        for i in range(3):
            y = self.finger_forces[i][start:]
            self.lines_touch[i].set_data(t_slice, y)
            if y:
                y_max_touch = max(y_max_touch, max(y) * 1.2)
        self.ax_touch.set_xlim(t_slice[0], t_slice[-1])
        self.ax_touch.set_ylim(0, max(y_max_touch, 0.5))

        # 중단 갱신
        y_total = self.total_force[start:]
        self.line_total.set_data(t_slice, y_total)
        self.ax_total.set_xlim(t_slice[0], t_slice[-1])
        y_max_total = max(max(y_total) * 1.2, 0.5) if y_total else 1.0
        self.ax_total.set_ylim(0, y_max_total)

        # 하단 갱신 (그래스프 분석)
        self.line_quality.set_data(t_slice, self.grasp_quality[start:])
        self.line_rank.set_data(t_slice, self.grasp_rank[start:])
        self.line_dof.set_data(t_slice, self.closed_chain_dof[start:])
        self.ax_grasp.set_xlim(t_slice[0], t_slice[-1])

        # Force closure 상태 텍스트
        fc = result['force_closure']
        nc = result['n_contacts']
        rank = result['G_rank']
        dof = result['closed_chain_dof']
        obj_dof = result['object_dof']
        fc_str = "FORCE CLOSURE" if fc else "NO CLOSURE"
        fc_color = '#27ae60' if fc else '#e74c3c'
        self.fc_text.set_text(
            f"{fc_str}\nContacts: {nc}  Rank(G): {rank}/6\n"
            f"Closed-chain DOF: {dof}  Obj DOF: {obj_dof}")
        self.fc_text.set_color(fc_color)

        # 드로우
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


# ═══════════════════════════════════════════════════════
#  메인
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    env = MujocoEnv()
    env.reset()

    q_home = env.get_arm_q()
    print(f"[Home] q = {np.rad2deg(q_home).round(1)}")
    print(f"[Home] EE pos = {env.get_ee_pos()}")
    print(f"[Home] Obj pos = {env.get_obj_pos()}")

    # ── 그래스핑 자세 계산 ──
    obj_pos = env.get_obj_pos()
    ee_home = env.get_ee_pos()
    home_z = ee_home[2]  # 초기 자세 높이

    # 초기 자세의 EE 회전을 그대로 사용
    R_grasp = env.get_ee_rot()

    # 박스 꼭대기 높이 (half-height 0.05, body pos z=0.0 → 꼭대기 z=0.05)
    obj_top_z = obj_pos[2] + 0.05

    # 1) approach: 초기 높이 유지하면서 물체 x,y 위로 이동
    p_approach = obj_pos.copy()
    p_approach[2] = home_z
    
    # 2) pre-grasp: 물체 바로 위로 하강
    p_pre_grasp = obj_pos.copy()
    p_pre_grasp[2] = obj_top_z + 0.3

    # 3) grasp: 원통 꼭대기에서 살짝 내려감 (그리퍼가 원통 상단을 감싸도록)
    p_grasp = obj_pos.copy()
    p_grasp[2] = obj_top_z + 0.1

    # 4) lift: 초기 높이까지 들어올림
    p_lift = obj_pos.copy()
    p_lift[2] = home_z

    # 5) place_above: 초기 높이 유지하면서 place 위치로 이동
    p_place_above = obj_pos.copy()
    p_place_above[0] -= 0.25
    p_place_above[1] += 0.25
    p_place_above[2] = home_z

    # 6) place: 내려놓기
    p_place = p_place_above.copy()
    p_place[2] = obj_top_z + 0.1

    # 7) retreat: 내려놓은 자리에서 z축으로만 올라가기
    p_retreat = p_place.copy()
    p_retreat[2] = home_z

    print(f"\n[목표] approach:    {p_approach}")
    print(f"[목표] pre-grasp:   {p_pre_grasp}")
    print(f"[목표] grasp:       {p_grasp}")
    print(f"[목표] lift:        {p_lift}")
    print(f"[목표] place_above: {p_place_above}")
    print(f"[목표] place:       {p_place}")
    print(f"[목표] retreat:     {p_retreat}")

    # ── IK 풀기 ──
    print("\n--- IK 풀기 ---")
    targets = [
        ("approach",    p_approach,    R_grasp),
        ("pre-grasp",   p_pre_grasp,   R_grasp),
        ("grasp",       p_grasp,       R_grasp),
        ("lift",        p_lift,        R_grasp),
        ("place_above", p_place_above, R_grasp),
        ("place",       p_place,       R_grasp),
        ("retreat",     p_retreat,     R_grasp),
    ]

    q_solutions = [q_home]
    q_prev = q_home.copy()

    for name, p_tgt, R_tgt in targets:
        q_sol, converged = env.solve_ik(p_tgt, R_tgt, q_prev)
        status = "OK" if converged else "FAIL"
        ee_pos = env.get_ee_pos()
        err = np.linalg.norm(p_tgt - ee_pos)
        print(f"  [{name:12s}] {status}  err={err*1000:.2f}mm  q={np.rad2deg(q_sol).round(1)}")
        q_solutions.append(q_sol)
        q_prev = q_sol.copy()

    # home으로 복귀
    q_solutions.append(q_home)

    # ── 궤적 생성 ──
    # 구간: home → approach → pre-grasp → grasp → (close) → lift → place_above → place → (open) → home
    traj = interp_joint_traj(q_solutions)
    print(f"\n[궤적] {len(traj)} steps ({len(traj)*SIM_DT:.1f}s)")

    # grasp/place 시점 계산 (대략적)
    seg_lengths = []
    for i in range(len(q_solutions) - 1):
        dist = np.max(np.abs(q_solutions[i+1] - q_solutions[i]))
        dur = max(dist / TRAJ_VEL, 0.5)
        seg_lengths.append(max(int(dur / SIM_DT), 2))

    cumulative = np.cumsum(seg_lengths)
    grasp_idx = cumulative[2]    # grasp 도달 시점 (pre-grasp → grasp 끝)
    place_idx = cumulative[5]    # place 도달 시점 (place_above → place 끝)

    print(f"[그리퍼] close at step {grasp_idx}, open at step {place_idx}")

    # ═══════════════════════════════════════════════════════
    #  시뮬레이션 실행
    # ═══════════════════════════════════════════════════════

    env.reset()
    render_every = max(1, int(1 / (RENDER_HZ * SIM_DT)))

    # 구간 경계 tick 집합 (hold할 시점)
    waypoint_ticks = set(cumulative.tolist())

    # 웨이포인트 좌표 목록 (시각화용)
    waypoint_positions = [p_tgt for (_, p_tgt, _) in targets]
    waypoint_colors = [
        [0, 1, 0, 0.6],    # approach: 초록
        [0, 0.7, 1, 0.6],  # pre-grasp: 하늘
        [1, 0, 0, 0.8],    # grasp: 빨강
        [1, 1, 0, 0.6],    # lift: 노랑
        [0, 0.7, 1, 0.6],  # place_above: 하늘
        [1, 0.5, 0, 0.8],  # place: 주황
        [0.5, 0, 1, 0.6],  # retreat: 보라
    ]
    waypoint_labels = ["approach", "pre-grasp", "grasp", "lift", "place_above", "place", "retreat"]

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True


        # 실시간 접촉력 그래프 창
        contact_plotter = ContactPlotter(env.model, env.data)
        plot_every = render_every * 5  # 그래프는 뷰어보다 덜 자주 갱신

        # 웨이포인트 구체 시각화
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
        waypoint_idx = 0  # 다음 웨이포인트 인덱스

        while viewer.is_running() and tick < len(traj):
            step_start = time.time()

            # 현재 목표 관절각
            q_target = traj[tick]

            # 그리퍼 상태 결정
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

            # 제어 입력 세팅
            env.data.ctrl[env.arm_act_ids] = q_target
            env.data.ctrl[env.grip_act_ids] = grip_cmd

            # 시뮬레이션 스텝
            mujoco.mj_step(env.model, env.data)

            # 렌더링 (+ 접촉 시각화 갱신)
            if tick % render_every == 0:
                # 웨이포인트 geom 수
                n_wp = len(waypoint_positions)
                # 접촉점 geom 수집
                contact_geoms = get_contact_geoms(env.model, env.data)
                # user_scn에 접촉 geom 추가 (웨이포인트 뒤에)
                for j, cg in enumerate(contact_geoms):
                    idx = n_wp + j
                    if idx >= viewer.user_scn.maxgeom:
                        break
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[idx],
                        type=cg['type'],
                        size=cg['size'],
                        pos=cg['pos'].astype(np.float64),
                        mat=cg['mat'].astype(np.float64),
                        rgba=cg['rgba'],
                    )
                viewer.user_scn.ngeom = n_wp + min(len(contact_geoms), viewer.user_scn.maxgeom - n_wp)
                viewer.sync()

            # 접촉력 그래프 갱신
            if tick % plot_every == 0:
                contact_plotter.update()

            tick += 1

            # 실시간 페이싱
            elapsed = time.time() - step_start
            sleep_time = SIM_DT - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            # 웨이포인트 도달 시 hold
            if waypoint_idx < len(cumulative) and tick == cumulative[waypoint_idx]:
                delay = WAYPOINT_DELAYS[waypoint_idx] if waypoint_idx < len(WAYPOINT_DELAYS) else 0.5
                print(f"[t={tick*SIM_DT:.2f}s] 웨이포인트 {waypoint_idx} 도달, {delay}초 hold")
                hold_remaining = delay
                while viewer.is_running() and hold_remaining > 0:
                    env.data.ctrl[env.arm_act_ids] = q_target
                    env.data.ctrl[env.grip_act_ids] = grip_cmd
                    mujoco.mj_step(env.model, env.data)
                    viewer.sync()
                    time.sleep(SIM_DT)
                    hold_remaining -= SIM_DT
                waypoint_idx += 1

        # 시뮬 끝난 후 잠시 대기
        print("\n[완료] Pick & Place 시뮬레이션 종료")
        remaining = 3.0  # 3초 대기
        while viewer.is_running() and remaining > 0:
            mujoco.mj_step(env.model, env.data)
            viewer.sync()
            time.sleep(SIM_DT)
            remaining -= SIM_DT

    print("[종료]")
