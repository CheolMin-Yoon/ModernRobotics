# -*- coding: utf-8 -*-
"""
ch07 폐연쇄 기구학: Cassie 로봇 (mujoco_menagerie) 테스트
MuJoCo로 Cassie 로드 → 폐연쇄 구속 조건 확인 → 4절 링크 분석

Cassie 폐연쇄 구조:
  - left-achilles-rod ↔ left-heel-spring  (connect 구속)
  - left-plantar-rod  ↔ left-foot         (connect 구속)
  - 오른쪽도 동일하게 2개
  총 4개의 equality connect 구속으로 루프를 닫음
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import mujoco

np.set_printoptions(precision=4, suppress=True)

CASSIE_DIR = os.path.join(os.path.dirname(__file__), '..',
    'mujoco_menagerie/agility_cassie')
CASSIE_XML = os.path.join(CASSIE_DIR, 'scene.xml')

# ── MuJoCo 모델 로드 ──
model = mujoco.MjModel.from_xml_path(CASSIE_XML)
data = mujoco.MjData(model)

print("=" * 60)
print("  ch07 폐연쇄: Cassie 로봇 모델 정보")
print("=" * 60)

# ── 모델 기본 정보 ──
print(f"\nnq (config dim)  : {model.nq}")
print(f"nv (velocity dim): {model.nv}")
print(f"nbody            : {model.nbody}")
print(f"njnt             : {model.njnt}")
print(f"nu (actuators)   : {model.nu}")
print(f"neq (constraints): {model.neq}")

# ── 조인트 정보 ──
print(f"\n--- 조인트 목록 ({model.njnt}개) ---")
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or f"(unnamed_{i})"
    jnt_type = model.jnt_type[i]
    type_name = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}[jnt_type]
    print(f"  [{i:2d}] {name:30s}  type={type_name}")

# ── 등식 구속 조건 (폐연쇄) ──
print(f"\n--- 등식 구속 조건 ({model.neq}개) ---")
for i in range(model.neq):
    eq_type = model.eq_type[i]
    type_name = {0: "connect", 1: "weld", 2: "joint", 3: "tendon", 4: "distance"}[eq_type]
    body1_id = model.eq_obj1id[i]
    body2_id = model.eq_obj2id[i]
    body1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1_id)
    body2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2_id)
    anchor = model.eq_data[i, :3]
    print(f"  [{i}] {type_name:8s}  {body1_name:25s} ↔ {body2_name:25s}  anchor={anchor}")

# ── home keyframe으로 초기화 ──
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

# ── 구속 위반 확인 ──
print(f"\n--- 구속 위반 (home config) ---")
print(f"  efc_pos (구속 위치 오차): {data.efc_pos[:model.neq*3]}")

# ── 왼쪽 다리 4절 링크 분석 ──
print(f"\n--- 왼쪽 다리 폐연쇄 body 위치 (home config) ---")
closed_chain_bodies = [
    "left-hip-pitch", "left-achilles-rod", "left-shin",
    "left-tarsus", "left-heel-spring", "left-foot-crank",
    "left-plantar-rod", "left-foot"
]
for name in closed_chain_bodies:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    pos = data.xpos[bid]
    print(f"  {name:25s}  pos={pos}")

# ── 액추에이터 정보 ──
print(f"\n--- 액추에이터 ({model.nu}개) ---")
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    print(f"  [{i:2d}] {name}")

print("\n" + "=" * 60)
print("  done — 모델 정보 출력 완료")
print("=" * 60)


# ── MuJoCo 뷰어로 Cassie 스폰 ──
def launch_viewer():
    """launch_passive로 Cassie를 스폰하고 시뮬레이션 실행"""
    import time

    print("\nMuJoCo 뷰어를 실행합니다. Cassie를 home config에서 스폰합니다.")
    print("뷰어 창을 닫으면 종료됩니다.\n")

    # home keyframe으로 초기화
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        t_start = time.time()
        while viewer.is_running():
            step_start = time.time()

            # 시뮬레이션 스텝
            mujoco.mj_step(model, data)
            viewer.sync()

            # 실시간 동기화
            dt = model.opt.timestep - (time.time() - step_start)
            if dt > 0:
                time.sleep(dt)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--view', action='store_true', help='MuJoCo 뷰어로 Cassie 스폰')
    args = parser.parse_args()

    if args.view:
        import mujoco.viewer
        launch_viewer()
