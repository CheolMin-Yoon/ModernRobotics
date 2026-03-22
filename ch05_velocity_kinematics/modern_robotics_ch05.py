__all__ = [
    'BodyJacobian', 'SpaceJacobian',
]

# 5장 속도 기구학과 정역학

import numpy as np
import os, sys
from scipy.linalg import expm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ch03_rigid_body_motion.modern_robotics_ch03 import *
from ch04_forward_kinematics.modern_robotics_ch04 import *


# 5.1 물체 자코비안 J_b 계산
# J_b[:,i] = Ad(e^{-[Bn]θn} ... e^{-[Bi+1]θi+1}) * B_i
# 입력: Blist (6xn 벡터 리스트), thetalist (nx1)
# 출력: J_b (6xn)

def BodyJacobian(Blist_vec, thetalist):
    n = len(thetalist)
    J_b = np.zeros((6, n))

    # 마지막 열은 그냥 B_n 그대로
    J_b[:, n - 1] = Blist_vec[n - 1]

    # 뒤에서부터 누적: e^{-[Bn]θn}, e^{-[Bn]θn}*e^{-[Bn-1]θn-1}, ...
    T = np.eye(4)
    for i in range(n - 2, -1, -1):
        # e^{-[Bi+1]θi+1} 누적
        T = T @ expm(-Vec2se3(Blist_vec[i + 1]) * thetalist[i + 1])
        J_b[:, i] = Adjoint(T) @ Blist_vec[i]

    return J_b


# 5.2 공간 자코비안 J_s 계산
# J_s[:,i] = Ad(e^{[S1]θ1} ... e^{[Si-1]θi-1}) * S_i
# 입력: Slist (6xn 벡터 리스트), thetalist (nx1)
# 출력: J_s (6xn)

def SpaceJacobian(Slist_vec, thetalist):
    n = len(thetalist)
    J_s = np.zeros((6, n))

    # 첫 번째 열은 S_1 그대로
    J_s[:, 0] = Slist_vec[0]

    # 앞에서부터 누적: e^{[S1]θ1}, e^{[S1]θ1}*e^{[S2]θ2}, ...
    T = np.eye(4)
    for i in range(1, n):
        # e^{[Si-1]θi-1} 누적
        T = T @ expm(Vec2se3(Slist_vec[i - 1]) * thetalist[i - 1])
        J_s[:, i] = Adjoint(T) @ Slist_vec[i]

    return J_s


if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True)

    print("=== UR5 자코비안 (θ2=-90°, θ5=90°) ===\n")

    J_b = BodyJacobian(Blist_body_vec, thetalist)
    J_s = SpaceJacobian(Slist_space_vec, thetalist)

    print("물체 자코비안 J_b:")
    print(J_b)
    print()
    print("공간 자코비안 J_s:")
    print(J_s)

    # 검증: J_s = [Ad_Tsb] * J_b
    from ch04_forward_kinematics.modern_robotics_ch04 import *
    T_sb = body_frame_fk(Blist_body, thetalist)
    Ad_Tsb = Adjoint(T_sb)
    J_s_from_Jb = Ad_Tsb @ J_b

    print("\n=== 검증: [Ad_Tsb] * J_b == J_s ===")
    print("Ad_Tsb @ J_b:")
    print(J_s_from_Jb)
    print(f"\n일치 여부: {np.allclose(J_s, J_s_from_Jb)}")
