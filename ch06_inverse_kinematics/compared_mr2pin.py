# -*- coding: utf-8 -*-
"""
ch06 IK: MR 뉴턴-랩슨 vs Pinocchio IK (URDF UR5e) 비교 검증
conda env: mr
python ch06_inverse_kinematics/compared_mr2pin.py
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pinocchio as pin
from pin_utils.pin_utils import *
from ch03_rigid_body_motion.modern_robotics_ch03 import *
from ch04_forward_kinematics.modern_robotics_ch04_ur5e import body_frame_fk, fixed_frame_fk
from ch05_velocity_kinematics.modern_robotics_ch05 import BodyJacobian, SpaceJacobian
from ch06_inverse_kinematics.modern_robotics_ch06 import *
from params.ur5e import *  # noqa: F403

np.set_printoptions(precision=6, suppress=True)

UR5E_URDF = URDF_PATH  # from params.ur5e

model, data = load_urdf(UR5E_URDF)

# ── Zero config에서 M, 스크류 추출 (Pinocchio 기준) ──
q_zero = np.zeros(model.nq)
M_pin = pin_fk(model, data, q_zero, 'tool0')

pin.forwardKinematics(model, data, q_zero)
pin.computeJointJacobians(model, data, q_zero)
Slist_pin_vec = []
for i in range(1, model.njoints):
    J = pin.getJointJacobian(model, data, i, pin.ReferenceFrame.WORLD)
    S_col = J[:, i - 1]
    w = S_col[3:].copy()
    v = S_col[:3].copy()
    Slist_pin_vec.append(np.concatenate([w, v]))

M_pin_inv = TransInv(M_pin)
Ad_Minv_pin = Adjoint(M_pin_inv)
Blist_pin_vec = [Ad_Minv_pin @ S for S in Slist_pin_vec]


# ── 테스트 ──
print("=" * 60)
print("  ch06 IK: MR (Newton-Raphson) vs Pinocchio (UR5e)")
print("=" * 60)

test_configs = {
    "theta2=-90, theta5=90": np.array([0, -np.pi/2, 0, 0, np.pi/2, 0]),
    "all 45 deg": np.full(6, np.pi/4),
    "random config 1": np.array([0.3, -1.2, 0.8, -0.5, 1.1, -0.7]),
    "random config 2": np.array([-0.5, -1.0, 1.5, -1.0, 0.5, 0.3]),
}

for name, q_target in test_configs.items():
    print(f"\n[{name}]  q_target = {np.round(q_target, 4)}")

    # 목표 pose (Pinocchio FK로 생성)
    T_sd = pin_fk(model, data, q_target, 'tool0')

    # 초기값: 영 위치
    q0 = np.zeros(6)

    # MR Body IK
    q_mr_b, ok_mr_b = IKinBody(Blist_pin_vec, M_pin, T_sd, q0)
    # MR Space IK
    q_mr_s, ok_mr_s = IKinSpace(Slist_pin_vec, M_pin, T_sd, q0)
    # Pinocchio IK
    q_pin, ok_pin, err_pin = pin_ik(model, data, T_sd, 'tool0', q0=q0)

    # FK 검증
    T_mr_b = pin_fk(model, data, q_mr_b, 'tool0') if ok_mr_b else None
    T_mr_s = pin_fk(model, data, q_mr_s, 'tool0') if ok_mr_s else None
    T_pin = pin_fk(model, data, q_pin, 'tool0') if ok_pin else None

    print(f"  MR Body IK:  수렴={ok_mr_b}", end="")
    if ok_mr_b:
        pose_err = np.linalg.norm(T_mr_b - T_sd)
        print(f"  pose_err={pose_err:.2e}")
    else:
        print()

    print(f"  MR Space IK: 수렴={ok_mr_s}", end="")
    if ok_mr_s:
        pose_err = np.linalg.norm(T_mr_s - T_sd)
        print(f"  pose_err={pose_err:.2e}")
    else:
        print()

    print(f"  Pinocchio IK: 수렴={ok_pin}", end="")
    if ok_pin:
        pose_err = np.linalg.norm(T_pin - T_sd)
        print(f"  pose_err={pose_err:.2e}  pin_err={err_pin:.2e}")
    else:
        print()

    # 해가 다를 수 있지만 (다중 해), FK 결과는 일치해야 함
    if ok_mr_b and ok_pin:
        compare("Body IK FK vs target", T_mr_b, T_sd, tol=1e-3)
    if ok_mr_s and ok_pin:
        compare("Space IK FK vs target", T_mr_s, T_sd, tol=1e-3)

print("\n" + "=" * 60)
print("  done")
print("=" * 60)
