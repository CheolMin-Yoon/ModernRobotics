# -*- coding: utf-8 -*-
"""
ch04 FK: PoE 직접 구현 vs MuJoCo (UR5e) 비교 검증
MuJoCo에서 UR5e를 로드하고, 다양한 관절 설정에서 FK 결과를 비교한다.

conda env: mr
python ch04_forward_kinematics/compared_mr2mujoco.py
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import mujoco
import mujoco.viewer

from ch04_forward_kinematics.modern_robotics_ch04 import (
    body_frame_fk, fixed_frame_fk,
)
from ch03_rigid_body_motion.modern_robotics_ch03 import (
    Adjoint, Vec2se3, TransInv
)

np.set_printoptions(precision=6, suppress=True)

# ── MuJoCo 모델 로드 ──
SCENE_XML = os.path.join(os.path.dirname(__file__), '..',
                         'mujoco_menagerie/universal_robots_ur5e/scene.xml')

model = mujoco.MjModel.from_xml_path(SCENE_XML)
data = mujoco.MjData(model)

# 조인트 이름 확인
joint_names = [model.joint(i).name for i in range(model.njnt)]
print(f"Joints: {joint_names}")

# site id for EE
site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'attachment_site')
print(f"EE site id: {site_id}")


# ── MuJoCo에서 zero config의 EE pose 추출 → M 행렬 구성 ──
def get_mujoco_ee_pose(model, data):
    """MuJoCo site의 SE(3) 변환 행렬 반환"""
    mujoco.mj_forward(model, data)
    pos = data.site_xpos[site_id].copy()
    rot = data.site_xmat[site_id].copy().reshape(3, 3)
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = pos
    return T


def set_joint_angles(data, q):
    """6개 조인트 각도 설정"""
    for i in range(6):
        data.qpos[i] = q[i]


# ── Zero config에서 M 추출 ──
set_joint_angles(data, np.zeros(6))
M = get_mujoco_ee_pose(model, data)
print(f"\nM (zero config EE pose):\n{M}")


# ── 수치 미분으로 공간꼴 스크류 추출 ──
def extract_spatial_screws(model, data, eps=1e-7):
    """
    각 조인트를 미소 회전시켜 수치적으로 공간꼴 스크류 축 추출.
    MuJoCo의 조인트 축/위치 정보를 직접 사용.
    """
    screws = []
    for i in range(6):
        # MuJoCo 조인트 축 (world frame에서)
        # zero config에서의 축 방향
        set_joint_angles(data, np.zeros(6))
        mujoco.mj_forward(model, data)

        # 조인트 축: body frame에서 정의된 축을 world로 변환
        jnt = model.joint(i)
        body_id = jnt.bodyid[0]
        axis_local = jnt.axis.copy()

        # body의 world rotation
        body_rot = data.xmat[body_id].reshape(3, 3)
        w = body_rot @ axis_local
        w = w / np.linalg.norm(w)

        # 조인트 위치 (world frame)
        # anchor = body pos + body_rot @ jnt.pos
        body_pos = data.xpos[body_id].copy()
        q_point = body_pos + body_rot @ jnt.pos

        # v = -w x q
        v = -np.cross(w, q_point)
        screws.append(np.concatenate([w, v]))

    return screws


Slist_space_vec = extract_spatial_screws(model, data)
Slist_space = [Vec2se3(S) for S in Slist_space_vec]

# Body screw
M_inv = TransInv(M)
Ad_Minv = Adjoint(M_inv)
Blist_body_vec = [Ad_Minv @ S for S in Slist_space_vec]
Blist_body = [Vec2se3(B) for B in Blist_body_vec]

print("\nSpatial screws (from MuJoCo):")
for i, S in enumerate(Slist_space_vec):
    print(f"  S{i+1}: {S}")


# ── 비교 함수 ──
def compare(name, my_result, mj_result, tol=1e-4):
    diff = np.linalg.norm(np.asarray(my_result) - np.asarray(mj_result))
    status = "PASS" if diff < tol else "FAIL"
    print(f"  [{name}] {status}  (diff={diff:.2e})")
    if diff >= tol:
        print(f"    PoE :\n{np.asarray(my_result)}")
        print(f"    MuJoCo:\n{np.asarray(mj_result)}")


# ── 테스트 ──
print("\n" + "=" * 60)
print("  ch04 FK: PoE vs MuJoCo (UR5e)")
print("=" * 60)

test_configs = {
    "zero config": np.zeros(6),
    "theta2=-90, theta5=90": np.array([0, np.pi/2, 0, np.pi/2, np.pi/2, 0]),
    "all 45 deg": np.full(6, np.pi/4),
    "random config": np.array([0.3, -1.2, 0.8, -0.5, 1.1, -0.7]),
    "home keyframe": np.array([-np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0]),
}

for name, q in test_configs.items():
    print(f"\n[{name}]  thetalist = {np.round(q, 4)}")

    # MuJoCo FK
    set_joint_angles(data, q)
    T_mj = get_mujoco_ee_pose(model, data)

    # PoE FK
    T_space = fixed_frame_fk(Slist_space, q, M)
    T_body = body_frame_fk(Blist_body, q, M)

    compare("space FK vs MuJoCo", T_space, T_mj)
    compare("body  FK vs MuJoCo", T_body, T_mj)
    compare("space vs body", T_space, T_body, tol=1e-10)

print("\n" + "=" * 60)
print("  done — FK 비교 완료")
print("=" * 60)


# ── MuJoCo 뷰어로 시각화 (선택) ──
def launch_viewer():
    """launch_passive로 뷰어를 열고, test_configs를 순회하며 자세 변경"""
    import time

    print("\nMuJoCo 뷰어를 실행합니다. 각 설정을 3초 간격으로 순회합니다.")
    print("뷰어 창을 닫으면 종료됩니다.\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for name, q in test_configs.items():
            if not viewer.is_running():
                break
            print(f"  → {name}: thetalist = {np.round(q, 4)}")
            set_joint_angles(data, q)
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(3)

        # 마지막 자세 유지하며 뷰어 열어둠
        print("\n  순회 완료. 뷰어 창을 닫으면 종료됩니다.")
        while viewer.is_running():
            time.sleep(0.1)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--view', action='store_true', help='MuJoCo 뷰어 실행')
    args = parser.parse_args()

    if args.view:
        launch_viewer()
