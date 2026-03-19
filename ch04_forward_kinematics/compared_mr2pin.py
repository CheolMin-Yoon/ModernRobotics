# -*- coding: utf-8 -*-
"""
ch04 FK 직접 구현 (PoE) vs Pinocchio (URDF UR5) 비교 검증
conda env: mr
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pinocchio as pin
from ch04_forward_kinematics.modern_robotics_ch04 import body_frame_fk, fixed_frame_fk
from ch03_rigid_body_motion.modern_robotics_ch03 import (
    Adjoint, Vec2so3, Vec2se3, TransInv
)

np.set_printoptions(precision=6, suppress=True)

UR5_URDF = os.path.join(os.path.dirname(__file__), '..',
    'urdf_files_dataset/urdf_files/ros-industrial/xacro_generated/'
    'universal_robots/ur_description/urdf/ur5.urdf')

# ── Pinocchio 모델 로드 ──
model = pin.buildModelFromUrdf(UR5_URDF)
data = model.createData()

# tool0 프레임 ID
TOOL_FRAME = model.getFrameId('tool0')


# ── Pinocchio zero config에서 M 추출 ──
q_zero = np.zeros(model.nq)
pin.forwardKinematics(model, data, q_zero)
pin.updateFramePlacements(model, data)
M_pin = data.oMf[TOOL_FRAME].homogeneous.copy()

# ── URDF 실제 파라미터로 MR PoE 구성 ──
# M에서 파라미터 역산 (미터 단위)
R_m = M_pin[:3, :3]
p_m = M_pin[:3, 3]

M = M_pin.copy()

# 공간꼴 스크류: Pinocchio Jacobian에서 추출
pin.computeJointJacobians(model, data, q_zero)
Slist_space_vec = []
for i in range(1, model.njoints):
    J = pin.getJointJacobian(model, data, i, pin.ReferenceFrame.WORLD)
    S_pin = J[:, i-1]  # [v, w] in pin convention
    w = S_pin[3:].copy()
    v = S_pin[:3].copy()
    Slist_space_vec.append(np.concatenate([w, v]))  # MR [w, v]

# se(3) 행렬로 변환
Slist_space = [Vec2se3(S) for S in Slist_space_vec]

# Body screw: B_i = Ad_{M^{-1}} * S_i
M_inv = TransInv(M)
Ad_Minv = Adjoint(M_inv)
Blist_body_vec = [Ad_Minv @ S for S in Slist_space_vec]
Blist_body = [Vec2se3(B) for B in Blist_body_vec]


# ── 비교 함수 ──
def compare(name, my_result, pin_result, tol=1e-4):
    diff = np.linalg.norm(np.asarray(my_result) - np.asarray(pin_result))
    status = "PASS" if diff < tol else "FAIL"
    print(f"  [{name}] {status}  (diff={diff:.2e})")
    if diff >= tol:
        print(f"    MR  :\n{np.asarray(my_result)}")
        print(f"    Pin :\n{np.asarray(pin_result)}")


# ── 테스트 ──
print("=" * 55)
print("  ch04 FK: MR (PoE) vs Pinocchio (URDF UR5)")
print("=" * 55)

# 테스트 관절 각도들
test_configs = {
    "zero config": np.zeros(6),
    "theta2=-90, theta5=90": np.array([0, -np.pi/2, 0, 0, np.pi/2, 0]),
    "all 45 deg": np.full(6, np.pi/4),
    "random config": np.array([0.3, -1.2, 0.8, -0.5, 1.1, -0.7]),
}

for name, q in test_configs.items():
    print(f"\n[{name}]  q = {np.round(q, 4)}")

    # Pinocchio FK
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    T_pin = data.oMf[TOOL_FRAME].homogeneous

    # MR FK
    T_space = fixed_frame_fk(Slist_space, q, M)
    T_body = body_frame_fk(Blist_body, q, M)

    compare("space FK vs Pin", T_space, T_pin)
    compare("body  FK vs Pin", T_body, T_pin)
    compare("space vs body", T_space, T_body, tol=1e-10)

print("\n" + "=" * 55)
print("  done")
print("=" * 55)
