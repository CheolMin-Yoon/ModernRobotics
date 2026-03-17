import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from ch02_configuration_space.topology import S1, visualize_T2_from_S1
from ch02_configuration_space.Explicit_Representation import explicit_representation_S1
from ch02_configuration_space.Implicit_Representation import implicit_representation_S1
from ch02_configuration_space.constraints import g_holonomic, check_pfaffian_constraint

def gruebler_formula(m, N, J, f_i):
    """
    Grübler 공식: 로봇의 자유도(DOF) 계산
    
    Parameters:
    - m: 공간의 차원 (평면: 3, 공간: 6)
    - N: 링크(body) 개수 (고정된 ground 제외)
    - J: 조인트 개수
    - f_i: 각 조인트의 자유도 합
    
    Returns:
    - dof: 시스템의 자유도
    
    공식: dof = m(N - 1 - J) + f_i
    """
    dof = m * (N - 1 - J) + f_i
    return dof


def visualize_2link_cspace():
    """2-링크 로봇의 C-space (T²) 시각화"""
    print("2-링크 로봇의 Configuration Space는 T² (토러스)입니다.")
    print("각 조인트가 S¹이고, S¹ × S¹ = T²")
    visualize_T2_from_S1()


if __name__ == "__main__":
    # 1. Grübler 공식 테스트
    print("=== Grübler 공식 예제 ===")
    print("평면 2-링크 로봇:")
    # m=3 (평면), N=2 (링크 2개), J=2 (조인트 2개), f_i=2 (각 조인트 1 DOF씩)
    dof = gruebler_formula(m=3, N=2, J=2, f_i=2)
    print(f"자유도: {dof}")
    print(f"계산: dof = m(N-1-J) + f_i = 3(2-1-2) + 2 = 3(-1) + 2 = -1")
    print("이건 잘못된 결과입니다. N은 ground 포함해야 합니다!\n")
    
    print("올바른 계산 (N=3, ground 포함):")
    dof_correct = gruebler_formula(m=3, N=3, J=2, f_i=2)
    print(f"자유도: {dof_correct}")
    print(f"계산: dof = 3(3-1-2) + 2 = 3(0) + 2 = 2 ✓\n")
    
    # 2. 명시적/암묵적 표현 비교
    print("=== S¹의 두 가지 표현 ===")
    explicit_representation_S1()
    implicit_representation_S1()
    
    # 3. 2-링크 로봇 C-space
    visualize_2link_cspace()