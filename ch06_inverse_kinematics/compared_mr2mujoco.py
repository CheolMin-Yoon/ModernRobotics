# -*- coding: utf-8 -*-
"""
ch06 IK: MR 뉴턴-랩슨 vs MuJoCo (UR5e) 비교 검증
MuJoCo에서 FK로 목표 pose를 생성하고, MR IK로 풀어서 검증한다.

conda env: mr
python ch06_inverse_kinematics/compared_mr2mujoco.py
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import mujoco
from ch03_rigid_body_motion.modern_robotics_ch03 import *
from ch04_forward_kinematics.modern_robotics_ch04_ur5e import body_frame_fk, fixed_frame_fk
from ch05_velocity_kinematics.modern_robotics_ch05 import BodyJacobian, SpaceJacobian
from ch06_inverse_kinematics.modern_robotics_ch06 import *
from params.ur5e import *  # noqa: F403

np.set_printoptions(precision=6, suppress=True)

# ── MuJoCo 모델 로드 ──
SCENE_XML = MUJOCO_SCENE  # from params.ur5e

mjmodel = mujoco.MjModel.from_xml_path(SCENE_XML)
mjdata = mujoco.MjData(mjmodel)

site_id = mujoco.mj_name2id(mjmodel, mujoco.mjtObj.mjOBJ_SITE, 'attachment_site')


def set_joint_angles(data, q):
    for i in range(6):
        data.qpos[i] = q[i]


def get_ee_pose(model, data):
    mujoco.mj_forward(model, data)
    pos = data.site_xpos[site_id].copy()
    rot = data.site_xmat[site_id].reshape(3, 3).copy()
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = pos
    return T


# ── Zero config에서 M, 스크류 추출 ──
set_joint_angles(mjdata, np.zeros(6))
M_mj = get_ee_pose(mjmodel, mjdata)


def extract_spatial_screws(model, data):
    screws = []
    set_joint_angles(data, np.zeros(6))
    mujoco.mj_forward(model, data)
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
        screws.append(np.concatenate([w, v]))
    return screws


Slist_mj_vec = extract_spatial_screws(mjmodel, mjdata)
Slist_mj = [Vec2se3(S) for S in Slist_mj_vec]

M_mj_inv = TransInv(M_mj)
Ad_Minv_mj = Adjoint(M_mj_inv)
Blist_mj_vec = [Ad_Minv_mj @ S for S in Slist_mj_vec]
Blist_mj = [Vec2se3(B) for B in Blist_mj_vec]


# ── 비교 함수 ──
def compare(name, my_result, mj_result, tol=1e-3):
    diff = np.linalg.norm(np.asarray(my_result) - np.asarray(mj_result))
    status = "PASS" if diff < tol else "FAIL"
    print(f"  [{name}] {status}  (diff={diff:.2e})")
    if diff >= tol:
        print(f"    MR    :\n{np.asarray(my_result)}")
        print(f"    MuJoCo:\n{np.asarray(mj_result)}")


# ── 테스트 ──
print("=" * 60)
print("  ch06 IK: MR (Newton-Raphson) vs MuJoCo FK 검증 (UR5e)")
print("=" * 60)

test_configs = {
    "theta2=-90, theta5=90": np.array([0, -np.pi/2, 0, 0, np.pi/2, 0]),
    "theta2=90, theta4=90, theta5=90": np.array([0, np.pi/2, 0, np.pi/2, np.pi/2, 0]),
    "all 45 deg": np.full(6, np.pi/4),
    "random config": np.array([0.3, -1.2, 0.8, -0.5, 1.1, -0.7]),
    "home keyframe": np.array([-np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0]),
}

for name, q_target in test_configs.items():
    print(f"\n[{name}]  q_target = {np.round(q_target, 4)}")

    # MuJoCo FK로 목표 pose 생성
    set_joint_angles(mjdata, q_target)
    T_sd = get_ee_pose(mjmodel, mjdata)

    # 초기값: 영 위치
    q0 = np.zeros(6)

    # MR Body IK (MuJoCo 스크류 사용)
    q_sol_b, ok_b = IKinBody(Blist_mj_vec, M_mj, T_sd, q0)
    # MR Space IK
    q_sol_s, ok_s = IKinSpace(Slist_mj_vec, M_mj, T_sd, q0)

    print(f"  Body IK:  수렴={ok_b}")
    if ok_b:
        # IK 해로 MuJoCo FK 검증
        set_joint_angles(mjdata, q_sol_b)
        T_check_b = get_ee_pose(mjmodel, mjdata)
        compare("Body IK FK vs target", T_check_b, T_sd)

    print(f"  Space IK: 수렴={ok_s}")
    if ok_s:
        set_joint_angles(mjdata, q_sol_s)
        T_check_s = get_ee_pose(mjmodel, mjdata)
        compare("Space IK FK vs target", T_check_s, T_sd)

print("\n" + "=" * 60)
print("  done")
print("=" * 60)
