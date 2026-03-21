# -*- coding: utf-8 -*-
"""
ch03 직접 구현 함수 vs Pinocchio 비교 검증
conda env: mr
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from pin_utils.pin_utils import *
from ch03_rigid_body_motion.modern_robotics_ch03 import *

np.set_printoptions(precision=6, suppress=True)


def mr2pin_twist(V_mr):
    """MR twist [w,v] -> Pinocchio twist [v,w]"""
    return np.concatenate([V_mr[3:], V_mr[:3]])


def pin2mr_twist(V_pin):
    """Pinocchio twist [v,w] -> MR twist [w,v]"""
    return np.concatenate([V_pin[3:], V_pin[:3]])


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
print("  ch03 MR vs Pinocchio (via pin_utils)")
print("=" * 55)

# 1. MatrixExp3 vs pin_exp3
print("\n[1] MatrixExp3 vs pin_exp3")
compare("exp3 (SO3)", MatrixExp3(hat_w, theta), pin_exp3(omega))

# 2. MatrixLog3 vs pin_log3
print("\n[2] MatrixLog3 vs pin_log3")
my_log3_vec = so32Vec(MatrixLog3(R_test))
compare("log3 (so3)", my_log3_vec, pin_log3(R_test))

# 3. MatrixExp6 vs pin_exp6
print("\n[3] MatrixExp6 vs pin_exp6")
se3mat = Vec2se3(twist6)
pin_twist6 = mr2pin_twist(twist6)
compare("exp6 (SE3)", MatrixExp6(se3mat), pin_exp6(pin_twist6))

# 4. MatrixLog vs pin_log6
print("\n[4] MatrixLog vs pin_log6")
my_log6_vec = se32Vec(MatrixLog(T_test))
pin_log6_mr = pin2mr_twist(pin_log6(T_test))
compare("log6 (se3)", my_log6_vec, pin_log6_mr)

# 5. TransInv roundtrip
print("\n[5] TransInv roundtrip")
compare("TransInv", TransInv(T_test) @ T_test, np.eye(4))

# 6. Rp2Trans <-> Trans2Rp roundtrip
print("\n[6] Rp2Trans <-> Trans2Rp roundtrip")
R_orig, p_orig = Trans2Rp(T_test)
compare("roundtrip", Rp2Trans(R_orig, p_orig), T_test)

# 7. AxisAng3
print("\n[7] AxisAng3 rebuild")
hat_w_out, theta_out = AxisAng3(omega)
compare("AxisAng3", hat_w_out * theta_out, omega)

# 8. AxisAng (6D)
print("\n[8] AxisAng 6D rebuild")
S_out, theta_out = AxisAng(twist6)
compare("AxisAng6", S_out * theta_out, twist6)

print("\n" + "=" * 55)
print("  done")
print("=" * 55)
