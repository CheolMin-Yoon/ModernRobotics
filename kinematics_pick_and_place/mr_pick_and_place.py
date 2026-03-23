"""
Modern Robotics (ch02~08) IK + MuJoCo 시뮬 기반 Pick & Place
- FK/IK/Jacobian: ch03~06 + ch08 (osqp_ik.lagrangian_ik)
- 물리 시뮬 + 렌더링: MuJoCo
- 궤적: 관절 공간 직선 보간 (등속)

실행:
  python3 kinematics_pick_and_place/mr_pick_and_place.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import mujoco
import mujoco.viewer
import time
import matplotlib
matplotlib.use('TkAgg')

from config import *
from sim_common import MujocoSim, interp_joint_traj, ContactPlotter
from osqp_ik import osqp_ik
from ch03_rigid_body_motion.modern_robotics_ch03 import *
from ch04_forward_kinematics.modern_robotics_ch04_ur5e import body_frame_fk, fixed_frame_fk
from params.ur5e import (
    Mlist, Glist,
    q_lower, q_upper, dq_max, tau_max,
)

np.set_printoptions(precision=4, suppress=True)


# ═══════════════════════════════════════════════════════
#  MuJoCo scene에서 MR 파라미터 추출
#  (URDF params와 MJCF의 좌표 규약이 다르므로 직접 추출)
# ═══════════════════════════════════════════════════════

def extract_mr_params_from_mujoco(scene_xml, ee_body_name='gripper_palm'):
    """MuJoCo scene에서 M, Slist, Blist를 추출하여 MR FK == MuJoCo FK를 보장"""
    model = mujoco.MjModel.from_xml_path(scene_xml)
    data = mujoco.MjData(model)
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_body_name)

    # zero config에서 M 추출
    for i in range(6):
        data.qpos[i] = 0.0
    mujoco.mj_forward(model, data)

    M = np.eye(4)
    M[:3, :3] = data.xmat[ee_id].reshape(3, 3)
    M[:3, 3] = data.xpos[ee_id]

    # 공간꼴 스크류 축 추출
    Slist_vec = []
    for i in range(6):
        jnt = model.joint(i)
        body_id = jnt.bodyid[0]
        axis_local = jnt.axis.copy()
        body_rot = data.xmat[body_id].reshape(3, 3)
        w = body_rot @ axis_local
        w = w / np.linalg.norm(w)
        body_pos = data.xpos[body_id].copy()
        q_point = body_pos + body_rot @ jnt.pos
        v = -np.cross(w, q_point)
        Slist_vec.append(np.concatenate([w, v]))

    # 물체꼴 스크류 축
    M_inv = TransInv(M)
    Ad_Minv = Adjoint(M_inv)
    Blist_vec = [Ad_Minv @ S for S in Slist_vec]
    Blist = [Vec2se3(B) for B in Blist_vec]
    Slist = [Vec2se3(S) for S in Slist_vec]

    return M, Slist, Slist_vec, Blist, Blist_vec


M_e, Slist_space, Slist_space_vec, Blist_body, Blist_body_vec = \
    extract_mr_params_from_mujoco(SCENE_XML, EE_BODY)


# ═══════════════════════════════════════════════════════
#  MR FK 헬퍼
# ═══════════════════════════════════════════════════════

def mr_fk(thetalist):
    """MR body-frame FK → (pos(3,), rot(3x3))"""
    T = body_frame_fk(Blist_body, thetalist, M_e)
    R, p = Trans2Rp(T)
    return p.flatten(), R


def mr_fk_T(thetalist):
    """MR body-frame FK → T(4x4)"""
    return body_frame_fk(Blist_body, thetalist, M_e)


def pose2T(R, p):
    """rot(3x3) + pos(3,) → T(4x4)"""
    return Rp2Trans(R, p)


# ═══════════════════════════════════════════════════════
#  메인
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    sim = MujocoSim()
    sim.reset()

    q_home = sim.get_arm_q()
    print(f"[Home] q = {np.rad2deg(q_home).round(1)}")

    # MR FK로 home EE 확인 (MuJoCo에서 추출한 파라미터이므로 MuJoCo world 좌표와 일치)
    p_home, R_home = mr_fk(q_home)
    ee_pos_mj = sim.get_ee_pos()
    print(f"[MR FK]     EE pos = {p_home}")
    print(f"[MuJoCo FK] EE pos = {ee_pos_mj}")

    # 물체 위치 (MuJoCo world == MR world)
    obj_pos = sim.get_obj_pos()
    print(f"[Obj pos] = {obj_pos}")

    # ── 웨이포인트 계산 (MuJoCo world 좌표계) ──
    R_grasp = R_home.copy()
    home_z = p_home[2]
    obj_top_z = obj_pos[2] + 0.05  # 박스 half-height

    p_approach    = obj_pos.copy();  p_approach[2] = home_z
    p_pre_grasp   = obj_pos.copy();  p_pre_grasp[2] = obj_top_z + 0.3
    p_grasp       = obj_pos.copy();  p_grasp[2] = obj_top_z + 0.1
    p_lift        = obj_pos.copy();  p_lift[2] = home_z

    p_place_above = obj_pos.copy()
    p_place_above[0] -= 0.25
    p_place_above[1] += 0.25
    p_place_above[2] = home_z

    p_place   = p_place_above.copy();  p_place[2] = obj_top_z + 0.1
    p_retreat = p_place.copy();        p_retreat[2] = home_z

    targets = [
        ("approach",    p_approach,    R_grasp),
        ("pre-grasp",   p_pre_grasp,   R_grasp),
        ("grasp",       p_grasp,       R_grasp),
        ("lift",        p_lift,        R_grasp),
        ("place_above", p_place_above, R_grasp),
        ("place",       p_place,       R_grasp),
        ("retreat",     p_retreat,     R_grasp),
    ]

    print("\n--- 웨이포인트 ---")
    for name, p, _ in targets:
        print(f"  {name:12s}: {p}")

    # ── MR OSQP IK 풀기 ──
    print("\n--- MR Lagrangian IK (OSQP) ---")
    q_solutions = [q_home]
    q_prev = q_home.copy()

    for name, p_tgt, R_tgt in targets:
        T_sd = pose2T(R_tgt, p_tgt)
        q_sol, converged = osqp_ik(
            Blist_body_vec, M_e, T_sd, q_prev,
            use_mass_matrix=False, max_iter=200,
        )
        p_fk, _ = mr_fk(q_sol)
        err = np.linalg.norm(p_tgt - p_fk)
        status = "OK" if converged else "FAIL"
        print(f"  [{name:12s}] {status}  err={err*1000:.2f}mm  q={np.rad2deg(q_sol).round(1)}")
        q_solutions.append(q_sol)
        q_prev = q_sol.copy()

    # home 복귀
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

    waypoint_positions = [p_tgt.copy() for (_, p_tgt, _) in targets]
    waypoint_colors = [
        [0, 1, 0, 0.6],    # approach: 초록
        [0, 0.7, 1, 0.6],  # pre-grasp: 하늘
        [1, 0, 0, 0.8],    # grasp: 빨강
        [1, 1, 0, 0.6],    # lift: 노랑
        [0, 0.7, 1, 0.6],  # place_above: 하늘
        [1, 0.5, 0, 0.8],  # place: 주황
        [0.5, 0, 1, 0.6],  # retreat: 보라
    ]

    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = False
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True

        contact_plotter = ContactPlotter(sim.model, sim.data)
        plot_every = render_every * 5

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
        waypoint_idx = 0

        while viewer.is_running() and tick < len(traj):
            step_start = time.time()
            q_target = traj[tick]

            # 그리퍼 상태
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

            # 제어 입력
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

            # 웨이포인트 도달 시 hold
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

        print("\n[완료] MR OSQP IK Pick & Place 시뮬레이션 종료")
        remaining = 3.0
        while viewer.is_running() and remaining > 0:
            mujoco.mj_step(sim.model, sim.data)
            viewer.sync()
            time.sleep(SIM_DT)
            remaining -= SIM_DT

    print("[종료]")
