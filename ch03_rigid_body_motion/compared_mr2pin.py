# -*- coding: utf-8 -*-
"""
ch03 직접 구현 함수 vs Pinocchio 비교 검증
conda env: mr
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pinocchio as pin
from ch03_rigid_body_motion.modern_robotics_ch03 import *

np.set_printoptions(precision=6, suppress=True)


def compare(name, my_result, pin_result, tol=1e-6):
    diff = np.linalg.norm(np.asarray(my_result) - np.asarray(pin_result))
    status = "PASS" if diff < tol else "FAIL"
    print(f"  [{name}] {status}  (diff={diff:.2e})")
    if diff >= tol:
        print(f"    my  : {np.asarray(my_result).flatten()[:6]}")
        print(f"    pin : {np.asarray(pin_result).flatten()[:6]}")


def mr2pin_twist(V_mr):
    """MR twist [w,v] -> Pinocchio twist [v,w]"""
    return np.concatenate([V_mr[3:], V_mr[:3]])


def pin2mr_twist(V_pin):
    """Pinocchio twist [v,w] -> MR twist [w,v]"""
    return np.concatenate([V_pin[3:], V_pin[:3]])


def mr2pin_adjoint(Ad_mr):
    """MR Adjoint [w;v] 순서 -> Pinocchio [v;w] 순서 변환"""
    # MR: [[R, 0], [pR, R]]  (w-first)
    # Pin: [[R, pR], [0, R]]  (v-first)
    # 행/열 순서 swap: [3:6, 0:3] <-> [0:3, 3:6]
    P = np.zeros((6, 6))
    P[:3, 3:] = np.eye(3)
    P[3:, :3] = np.eye(3)
    return P @ Ad_mr @ P


# -- test data --
omega = np.array([0.0, 0.2, 0.8])
theta = np.linalg.norm(omega)
hat_w = omega / theta

R_test = MatrixExp3(hat_w, theta)

T_test = np.array([
    [0, -1, 0, 0.1],
    [1,  0, 0, 0.2],
    [0,  0, 1, 0.3],
    [0,  0, 0, 1.0],
])

twist6 = np.array([0.0, 0.2, 0.8, 1.0, -0.5, 0.3])  # [w; v]


print("=" * 55)
print("  ch03 MR vs Pinocchio")
print("=" * 55)

# 1. Vec2so3 vs pin.skew
print("\n[1] Vec2so3 vs pin.skew")
compare("skew-symmetric", Vec2so3(omega), pin.skew(omega))

# 2. so32Vec vs pin.unSkew
print("\n[2] so32Vec vs pin.unSkew")
compare("unskew", so32Vec(Vec2so3(omega)), pin.unSkew(pin.skew(omega)))

# 3. MatrixExp3 vs pin.exp3
print("\n[3] MatrixExp3 vs pin.exp3")
compare("exp3 (SO3)", MatrixExp3(hat_w, theta), pin.exp3(omega))

# 4. MatrixLog3 vs pin.log3
print("\n[4] MatrixLog3 vs pin.log3")
my_log3_vec = so32Vec(MatrixLog3(R_test))
pin_log3_vec = pin.log3(R_test)
compare("log3 (so3)", my_log3_vec, pin_log3_vec)

# 5. MatrixExp6 vs pin.exp6
#    MR twist=[w,v], pin twist=[v,w] -> 변환 필요
print("\n[5] MatrixExp6 vs pin.exp6")
se3mat = Vec2se3(twist6)
pin_twist6 = mr2pin_twist(twist6)
compare("exp6 (SE3)", MatrixExp6(se3mat), pin.exp6(pin_twist6).homogeneous)

# 6. MatrixLog vs pin.log6
#    pin.log6 반환: [v,w] -> MR 순서 [w,v]로 변환
print("\n[6] MatrixLog vs pin.log6")
my_log6_vec = se32Vec(MatrixLog(T_test))
pin_log6_raw = pin.log6(pin.SE3(T_test)).vector
pin_log6_mr = pin2mr_twist(pin_log6_raw)
compare("log6 (se3)", my_log6_vec, pin_log6_mr)

# 7. TransInv vs SE3.inverse
print("\n[7] TransInv vs SE3.inverse")
compare("TransInv", TransInv(T_test), pin.SE3(T_test).inverse().homogeneous)

# 8. Adjoint vs SE3.action
#    MR [w;v] 순서 -> pin [v;w] 순서 변환
print("\n[8] Adjoint vs SE3.action")
my_Ad_pin_order = mr2pin_adjoint(Adjoint(T_test))
compare("Adjoint (6x6)", my_Ad_pin_order, pin.SE3(T_test).action)

# 9. Rp2Trans <-> Trans2Rp roundtrip
print("\n[9] Rp2Trans <-> Trans2Rp roundtrip")
R_orig, p_orig = Trans2Rp(T_test)
compare("roundtrip", Rp2Trans(R_orig, p_orig), T_test)

# 10. AxisAng3
print("\n[10] AxisAng3 rebuild")
hat_w_out, theta_out = AxisAng3(omega)
compare("AxisAng3", hat_w_out * theta_out, omega)

# 11. AxisAng (6D)
print("\n[11] AxisAng 6D rebuild")
S_out, theta_out = AxisAng(twist6)
compare("AxisAng6", S_out * theta_out, twist6)

print("\n" + "=" * 55)
print("  done")
print("=" * 55)
