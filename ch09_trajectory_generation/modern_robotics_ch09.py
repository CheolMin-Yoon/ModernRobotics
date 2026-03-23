"""
Modern Robotics Chapter 9: Trajectory Generation

Core trajectory generation functions from MR (Lynch & Park).
"""
import numpy as np


def quintic_trajectory(start_pos, start_vel, start_acc,
                       end_pos, end_vel, end_acc,
                       duration, num_points):
    """5차 다항식 궤적 생성 (MR Ch9.2)

    Args:
        start_pos, end_pos: 시작/끝 위치 (n_joints,)
        start_vel, end_vel: 시작/끝 속도 (n_joints,)
        start_acc, end_acc: 시작/끝 가속도 (n_joints,)
        duration: 총 시간 [s]
        num_points: 궤적 포인트 수

    Returns:
        t, positions, velocities, accelerations, jerks
    """
    n_joints = len(start_pos)
    t = np.linspace(0, duration, num_points)
    T = duration

    joint_coeffs = []
    for i in range(n_joints):
        A = np.array([
            [0,       0,       0,       0,     0, 1],
            [0,       0,       0,       0,     1, 0],
            [0,       0,       0,       2,     0, 0],
            [T**5,    T**4,    T**3,    T**2,  T, 1],
            [5*T**4,  4*T**3,  3*T**2,  2*T,   1, 0],
            [20*T**3, 12*T**2, 6*T,     2,     0, 0],
        ])
        b = np.array([start_pos[i], start_vel[i], start_acc[i],
                       end_pos[i],   end_vel[i],   end_acc[i]])
        x = np.linalg.solve(A, b)
        joint_coeffs.append(x)

    positions     = np.zeros((num_points, n_joints))
    velocities    = np.zeros((num_points, n_joints))
    accelerations = np.zeros((num_points, n_joints))
    jerks         = np.zeros((num_points, n_joints))

    for i in range(num_points):
        for j in range(n_joints):
            c = joint_coeffs[j]
            positions[i, j]     = np.polyval(c, t[i])
            velocities[i, j]    = np.polyval(np.polyder(c), t[i])
            accelerations[i, j] = np.polyval(np.polyder(c, 2), t[i])
            jerks[i, j]         = np.polyval(np.polyder(c, 3), t[i])

    return t, positions, velocities, accelerations, jerks


def trapezoidal_velocity_profile(d_total, v_max, a_max, n_points=1000):
    """사다리꼴 속도 프로파일 생성 (MR Ch9.3)"""
    t_ramp = v_max / a_max
    d_ramp = 0.5 * a_max * t_ramp**2

    if d_total <= 2 * d_ramp:
        t_ramp = np.sqrt(d_total / a_max)
        t_flat = 0
    else:
        t_flat = (d_total - 2 * d_ramp) / v_max

    t_total = 2 * t_ramp + t_flat
    t = np.linspace(0, t_total, n_points)

    position = np.zeros_like(t)
    velocity = np.zeros_like(t)
    acceleration = np.zeros_like(t)

    for i in range(len(t)):
        if t[i] < t_ramp:
            position[i] = 0.5 * a_max * t[i]**2
            velocity[i] = a_max * t[i]
            acceleration[i] = a_max
        elif t[i] < t_ramp + t_flat:
            position[i] = v_max * (t[i] - t_ramp / 2)
            velocity[i] = v_max
            acceleration[i] = 0
        else:
            position[i] = d_total - 0.5 * a_max * (t_total - t[i])**2
            velocity[i] = v_max - a_max * (t[i] - t_ramp - t_flat)
            acceleration[i] = -a_max

    return t, position, velocity, acceleration


def minimum_jerk_trajectory(q_init, q_final, total_time=1.0, dt=0.01):
    """최소 저크 궤적 생성"""
    T = total_time
    t_list, q_list = [], []
    t = 0
    while t < T:
        s = t / T
        q = q_init + (q_final - q_init) * (10*s**3 - 15*s**4 + 6*s**5)
        t_list.append(t)
        q_list.append(q)
        t += dt
    return np.array(t_list), np.array(q_list)


def trapezoidal_spline(q_init, q_final, max_velocity, max_acceleration,
                       n_points=1000):
    """단일 조인트 사다리꼴 스플라인"""
    delta = q_final - q_init
    t_accel = max_velocity / max_acceleration
    x_accel = 0.5 * max_acceleration * t_accel**2

    if abs(delta) >= 2 * x_accel:
        t_constant = (abs(delta) - 2 * x_accel) / max_velocity
    else:
        t_constant = 0
        t_accel = np.sqrt(abs(delta) / max_acceleration)

    t_total = 2 * t_accel + t_constant
    t = np.linspace(0, t_total, n_points)
    q = np.zeros_like(t)

    mask1 = t < t_accel
    mask2 = (t >= t_accel) & (t < t_accel + t_constant)
    mask3 = t >= t_accel + t_constant

    sign = 1.0 if q_final > q_init else -1.0
    q[mask1] = q_init + sign * 0.5 * max_acceleration * t[mask1]**2
    q[mask2] = q_init + sign * (x_accel + max_velocity * (t[mask2] - t_accel))
    q[mask3] = q_final - sign * 0.5 * max_acceleration * (t_total - t[mask3])**2

    return q, t


def s_curve_velocity_profile(t_total=1.0, t_ramp=0.3, v_max=1.0,
                             n_per_seg=100):
    """S-커브 속도 프로파일 생성"""
    a_max = v_max / t_ramp

    t1 = np.linspace(0, t_ramp, n_per_seg)
    t2 = np.linspace(t_ramp, t_total - t_ramp, n_per_seg)
    t3 = np.linspace(t_total - t_ramp, t_total, n_per_seg)

    v1 = a_max * t1
    v2 = np.ones_like(t2) * v_max
    v3 = v_max - a_max * (t3 - (t_total - t_ramp))

    t = np.concatenate((t1, t2, t3))
    v = np.concatenate((v1, v2, v3))

    dt = t[1] - t[0]
    p = np.cumsum(v) * dt
    a = np.gradient(v, dt)
    j = np.gradient(a, dt)

    return t, p, v, a, j


def linear_trajectory(start_pos, end_pos, num_points):
    """선형 보간 궤적 생성"""
    n_joints = len(start_pos)
    s = np.linspace(0, 1, num_points)
    trajectory = np.zeros((num_points, n_joints))
    for i in range(n_joints):
        trajectory[:, i] = start_pos[i] + s * (end_pos[i] - start_pos[i])
    return trajectory
