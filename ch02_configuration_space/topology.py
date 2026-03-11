import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# 기본 위상 공간 (Topological Spaces)
# ==========================================

def E(n):
    """
    유클리드 공간 E^n (Euclidean Space)
    n차원 실수 공간 R^n
    """
    return f"E^{n}: {n}차원 유클리드 공간"


def S1(angle, radius=1.0):
    """
    S^1: 1차원 원 (Circle)
    매개변수: angle ∈ [0, 2π)
    반환: (x, y) = (r*cos(θ), r*sin(θ))
    """
    return radius * np.cos(angle), radius * np.sin(angle)


def T(n):
    """
    n차원 토러스 T^n = S^1 × S^1 × ... × S^1 (n번)
    예: T^2 = S^1 × S^1 (도넛 모양)
        T^3 = S^1 × S^1 × S^1 (3-링크 로봇의 C-space)
    """
    return f"T^{n}: {n}차원 토러스 (S^1을 {n}번 곱한 공간)"


# ==========================================
# 시각화 함수들
# ==========================================

def visualize_T2_from_S1():
    """T^2 = S^1 × S^1 시각화 (2-링크 로봇)"""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    theta1 = np.linspace(0, 2*np.pi, 50)
    theta2 = np.linspace(0, 2*np.pi, 50)
    U, V = np.meshgrid(theta1, theta2)
    
    R = 2.0  # Major radius
    r = 1.0  # Minor radius

    x_small, z_small = S1(V, r)
    dir_x, dir_y = S1(U, 1.0)
    
    X = (R + x_small) * dir_x
    Y = (R + x_small) * dir_y
    Z = z_small

    ax.plot_surface(X, Y, Z, alpha=0.8, cmap='plasma', edgecolor='gray')
    ax.set_title("T² = S¹ × S¹ (2-Link Robot C-Space)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, r/R])
    plt.show()


def visualize_T3_projection():
    """
    T^3 = S^1 × S^1 × S^1 시각화 (3-링크 로봇)
    4차원 공간이므로 3차원으로 투영해서 표현
    각 축이 하나의 조인트 각도를 나타냄
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 3개의 조인트 각도
    n_samples = 20
    theta1 = np.linspace(0, 2*np.pi, n_samples)
    theta2 = np.linspace(0, 2*np.pi, n_samples)
    theta3 = np.linspace(0, 2*np.pi, n_samples)
    
    # 샘플링된 점들을 3D 공간에 표시
    # 실제로는 3차원 토러스를 완전히 시각화할 수 없지만,
    # 각 조인트의 주기성을 보여줄 수 있음
    for t1 in theta1[::4]:
        for t2 in theta2[::4]:
            x, y = S1(theta3, 1.0)
            z = np.ones_like(x) * t1
            ax.plot(x + t2/np.pi, y, z, 'b-', alpha=0.3, linewidth=0.5)
    
    ax.set_title("T³ = S¹ × S¹ × S¹ (3-Link Robot C-Space)\n투영된 표현")
    ax.set_xlabel("θ₁, θ₂ 방향")
    ax.set_ylabel("θ₃ 방향")
    ax.set_zlabel("높이")
    plt.show()


# ==========================================
# 사용 예제
# ==========================================

if __name__ == "__main__":
    print("=== 기본 위상 공간 ===")
    print(E(2))  # 평면
    print(E(3))  # 3차원 공간
    print(T(2))  # 2-링크 로봇
    print(T(3))  # 3-링크 로봇
    print()
    
    print("=== S^1 예제 ===")
    angles = np.array([0, np.pi/2, np.pi])
    for angle in angles:
        x, y = S1(angle)
        print(f"θ={angle:.2f}: ({x:.2f}, {y:.2f})")
    print()
    
    print("=== T^2 시각화 ===")
    visualize_T2_from_S1()
    
    print("=== T^3 시각화 ===")
    visualize_T3_projection()