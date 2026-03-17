import numpy as np

# ==========================================
# 1. 홀로노믹 제약조건 (Holonomic Constraint)
# ==========================================
def g_holonomic(theta):
    """
    예시: 2차원 평면의 점(x, y)이 반지름 R=5인 원 위에 있어야 함.
    수식: g(x, y) = x^2 + y^2 - R^2 = 0
    """
    x, y = theta
    R = 5.0
    constraint_value = x**2 + y**2 - R**2
    return constraint_value

# 테스트 (3, 4) 위치는 3^2 + 4^2 = 25 이므로 제약조건 만족(0)
theta_valid = np.array([3.0, 4.0])
print(f"홀로노믹 제약조건 평가 (3, 4): {g_holonomic(theta_valid)} (0이면 만족)")


# ==========================================
# 2. 파피안 제약조건 (Pfaffian Constraint)
# ==========================================
def A_pfaffian(theta):
    """
    예시: 외발 자전거(Unicycle) 로봇 (논홀로노믹의 대표적 예)
    상태변수: theta = [x, y, phi(헤딩 각도)]
    제약조건: 자동차는 옆으로 미끄러질 수 없다. (측면 속도 = 0)
    수식: x_dot * sin(phi) - y_dot * cos(phi) = 0
    A(theta) 행렬 = [sin(phi), -cos(phi), 0]
    """
    x, y, phi = theta
    return np.array([np.sin(phi), -np.cos(phi), 0.0])

def check_pfaffian_constraint(theta, theta_dot):
    """ A(theta) * theta_dot = 0 인지 확인 """
    A = A_pfaffian(theta)
    # 행렬과 벡터의 내적 (Dot product)
    violation = np.dot(A, theta_dot)
    return violation

# 테스트: 로봇이 45도(pi/4) 방향을 보고 있을 때
theta_robot = np.array([0.0, 0.0, np.pi/4])

# 속도 1: 앞으로 전진하는 속도 (올바른 움직임) -> 옆으로 안 미끄러짐
theta_dot_forward = np.array([np.cos(np.pi/4), np.sin(np.pi/4), 0.0])
print(f"\n파피안 제약조건 평가 (전진): {check_pfaffian_constraint(theta_robot, theta_dot_forward):.2f} (0이면 만족)")

# 속도 2: 옆으로 게걸음 치는 속도 (물리적으로 불가능한 움직임)
theta_dot_sideways = np.array([-np.sin(np.pi/4), np.cos(np.pi/4), 0.0])
print(f"파피안 제약조건 평가 (게걸음): {check_pfaffian_constraint(theta_robot, theta_dot_sideways):.2f} (0이 아니면 제약 위반!)")