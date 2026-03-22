# -*- coding: utf-8 -*-
"""ch04 정기구학 - UR5e (파라미터는 params.ur5e에서 참조)"""

__all__ = [
    'body_frame_fk', 'fixed_frame_fk',
]

import numpy as np
import os, sys
from scipy.linalg import expm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from params.ur5e import *  # noqa: F401,F403 — UR5e 파라미터


# ============================================================
# FK 함수
# ============================================================
def body_frame_fk(Blist_body, thetalist, M=M_e):
    T_body = np.copy(M)
    for B, theta in zip(Blist_body, thetalist):
        T_body = T_body @ expm(B * theta)
    return T_body


def fixed_frame_fk(Slist_space, thetalist, M=M_e):
    T_space = np.eye(4)
    for S, theta in zip(Slist_space, thetalist):
        T_space = T_space @ expm(S * theta)
    T_space = T_space @ M
    return T_space


if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)
    print("=== UR5e FK (θ2=-90°, θ5=90°) ===\n")

    thetalist_e = [0, -np.pi/2, 0, 0, np.pi/2, 0]

    print("M_e:")
    print(M_e)
    print()
    T_body = body_frame_fk(Blist_body, thetalist_e)
    T_space = fixed_frame_fk(Slist_space, thetalist_e)
    print("Body FK:")
    print(T_body)
    print()
    print("Space FK:")
    print(T_space)
    print()
    print(f"Body == Space: {np.allclose(T_body, T_space)}")
