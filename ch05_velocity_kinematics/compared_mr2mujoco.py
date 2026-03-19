# -*- coding: utf-8 -*-
"""
ch05 자코비안: MR 직접 구현 vs MuJoCo (UR5e) 비교 검증
conda env: mr
python ch05_velocity_kinematics/compared_mr2mujoco.py
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import mujoco
from ch03_rigid_body_motion.modern_robotics_ch03 import (
    Adjoint, Vec2se3, TransInv
)
from ch05_velocity_kinematics.modern_robotics_ch05 import (
    BodyJacobian, SpaceJacobian
)

np.set_printoptions(precision=6, suppress=True)

# ── MuJoCo 모델 로드 ──
SCENE_XML = os.path.join(os.path.dirname(__file__), '..',
                         'mujoco_menagerie/universal_robots_ur5e/scene.xml')

mjmodel = mujoco.MjModel.from_xml_path(SCENE_XML)
mjdata = mujoco.MjData(mjmodel)

site_id = mujoco.mj_name2id(mjmodel, mujoco.mjtObj.mjOBJ_SITE, 'attachment_site')
print(f"EE site id: {site_id}")


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


# ── Zero config에서 M, 스크류 추출 (ch04 mujoco 비교와 동일) ──
set_joint_angles(mjdata, np.zeros(6))
M = get_ee_pose(mjmodel, mjdata)
print(f"M (zero config):\n{M}")


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


Slist_space_vec = extract_spatial_screws(mjmodel, mjdata)
M_inv = TransInv(M)
Ad_Minv = Adjoint(M_inv)
Blist_body_vec = [Ad_Minv @ S for S in Slist_space_vec]


# ── MuJoCo 자코비안 (MR 컨벤션으로 변환) ──
def mujoco_jacobians(model, data, q):
    """
    MuJoCo site 자코비안을 MR [w,v] 컨벤션으로 변환.

    mj_jacSite의 jacp는 site 점의 선속도 자코비안:
        jacp_i = v_origin_i + w_i × p_site
    MR 공간 스크류의 v는 원점 기준이므로:
        v_origin_i = jacp_i - w_i × p_site
    """
    set_joint_angles(data, q)
    mujoco.mj_forward(model, data)

    jacp = np.zeros((3, model.nv))  # site 점 선속도 자코비안
    jacr = np.zeros((3, model.nv))  # 회전 자코비안
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)

    # site 위치 (world frame)
    p_site = data.site_xpos[site_id].copy()

    # 각 열에서 w_i × p_site 보정 → 원점 기준 v로 변환
    w_cols = jacr[:, :6]
    v_origin = jacp[:, :6] - np.cross(w_cols.T, p_site).T

    # MR [w, v] 순서로 조합
    J_s_mj = np.vstack([w_cols, v_origin])

    # 물체 자코비안: J_b = Ad(T_bs) @ J_s
    T_sb = get_ee_pose(model, data)
    T_bs = TransInv(T_sb)
    Ad_Tbs = Adjoint(T_bs)
    J_b_mj = Ad_Tbs @ J_s_mj

    return J_s_mj, J_b_mj


# ── 비교 함수 ──
def compare(name, mr_result, mj_result, tol=1e-4):
    diff = np.linalg.norm(np.asarray(mr_result) - np.asarray(mj_result))
    status = "PASS" if diff < tol else "FAIL"
    print(f"  [{name}] {status}  (diff={diff:.2e})")
    if diff >= tol:
        print(f"    MR    :\n{np.asarray(mr_result)}")
        print(f"    MuJoCo:\n{np.asarray(mj_result)}")


# ── 테스트 ──
print("\n" + "=" * 60)
print("  ch05 Jacobian: MR vs MuJoCo (UR5e)")
print("=" * 60)

test_configs = {
    "zero config": np.zeros(6),
    "theta2=90, theta5=90": np.array([0, np.pi/2, 0, np.pi/2, np.pi/2, 0]),
    "all 45 deg": np.full(6, np.pi/4),
    "random config": np.array([0.3, -1.2, 0.8, -0.5, 1.1, -0.7]),
    "home keyframe": np.array([-np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0]),
}

for name, q in test_configs.items():
    print(f"\n[{name}]  q = {np.round(q, 4)}")

    # MR 자코비안
    J_s_mr = SpaceJacobian(Slist_space_vec, q)
    J_b_mr = BodyJacobian(Blist_body_vec, q)

    # MuJoCo 자코비안
    J_s_mj, J_b_mj = mujoco_jacobians(mjmodel, mjdata, q)

    compare("Space Jacobian (J_s)", J_s_mr, J_s_mj)
    compare("Body  Jacobian (J_b)", J_b_mr, J_b_mj)

print("\n" + "=" * 60)
print("  done")
print("=" * 60)
