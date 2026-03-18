"""
Pinocchio 검증 유틸리티

직접 구현한 결과를 Pinocchio와 비교하는 헬퍼 함수들.
Pinocchio가 설치되어 있지 않으면 graceful하게 스킵.

챕터별 대응:
  ch03 — 강체 운동:     exp3/log3, exp6/log6, SE3, Quaternion
  ch04 — 정기구학:      forwardKinematics, updateFramePlacements
  ch05 — 속도 기구학:   Jacobian (joint/frame), dJ/dt
  ch06 — 역기구학:      IK loop (log6, Jlog6, integrate)
  ch08 — 동역학:        rnea, crba, aba, gravity, coriolis, energy
  ch09 — 궤적 생성:     FK waypoint 검증
  ch11 — 로봇 제어:     rnea 기반 CTC 검증, regressor
"""

import numpy as np
import os

try:
    import pinocchio as pin
    HAS_PINOCCHIO = True
except ImportError:
    HAS_PINOCCHIO = False
    print("[pin_utils] pinocchio not found — 검증 함수 비활성화")


# ═══════════════════════════════════════════════════════
#  모델 로드
# ═══════════════════════════════════════════════════════

def load_urdf(urdf_path):
    """URDF 로드 → (model, data)"""
    if not HAS_PINOCCHIO:
        return None, None
    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()
    return model, data


def model_info(model):
    """모델 기본 정보 출력"""
    if not HAS_PINOCCHIO:
        return
    print(f"  nq (config dim) : {model.nq}")
    print(f"  nv (velocity dim): {model.nv}")
    print(f"  njoints          : {model.njoints}")
    print(f"  nframes          : {model.nframes}")
    print(f"  joint names      : {[model.names[i] for i in range(model.njoints)]}")
    print(f"  q_min            : {model.lowerPositionLimit.T}")
    print(f"  q_max            : {model.upperPositionLimit.T}")


# ═══════════════════════════════════════════════════════
#  ch03: 강체 운동 — SO(3), SE(3), exp/log
# ═══════════════════════════════════════════════════════

def pin_exp3(omega):
    """so(3) → SO(3): Rodrigues formula.  omega = ω*θ (3-vector)"""
    if not HAS_PINOCCHIO:
        return None
    return pin.exp3(np.asarray(omega, dtype=float))


def pin_log3(R):
    """SO(3) → so(3): 회전 행렬 → 축-각 벡터 (3-vector)"""
    if not HAS_PINOCCHIO:
        return None
    return pin.log3(np.asarray(R, dtype=float))


def pin_exp6(twist):
    """se(3) → SE(3): 6D twist → 4x4 변환 행렬.  twist = [ω; v] (6-vector)"""
    if not HAS_PINOCCHIO:
        return None
    return pin.exp6(np.asarray(twist, dtype=float)).homogeneous


def pin_log6(T):
    """SE(3) → se(3): 4x4 변환 행렬 → 6D twist (Motion)"""
    if not HAS_PINOCCHIO:
        return None
    return pin.log6(np.asarray(T, dtype=float)).vector


def pin_Jexp3(omega):
    """exp3의 야코비안: d(exp3)/d(omega)"""
    if not HAS_PINOCCHIO:
        return None
    return pin.Jexp3(np.asarray(omega, dtype=float))


def pin_Jlog3(R):
    """log3의 야코비안: d(log3)/d(R)"""
    if not HAS_PINOCCHIO:
        return None
    return pin.Jlog3(np.asarray(R, dtype=float))


def pin_Jexp6(twist):
    """exp6의 야코비안"""
    if not HAS_PINOCCHIO:
        return None
    return pin.Jexp6(np.asarray(twist, dtype=float))


def pin_Jlog6(T):
    """log6의 야코비안"""
    if not HAS_PINOCCHIO:
        return None
    se3 = pin.SE3(np.asarray(T, dtype=float))
    return pin.Jlog6(se3)


# ═══════════════════════════════════════════════════════
#  ch04: 정기구학 — Forward Kinematics
# ═══════════════════════════════════════════════════════

def pin_fk(model, data, q, frame_name=None):
    """Pinocchio FK → 4x4 변환 행렬

    frame_name이 주어지면 해당 프레임, 아니면 마지막 조인트 프레임.
    """
    if not HAS_PINOCCHIO:
        return None
    pin.forwardKinematics(model, data, np.asarray(q, dtype=float))
    if frame_name:
        pin.updateFramePlacements(model, data)
        fid = model.getFrameId(frame_name)
        return data.oMf[fid].homogeneous.copy()
    else:
        return data.oMi[-1].homogeneous.copy()


def pin_fk_all(model, data, q):
    """모든 조인트의 FK 결과 반환 → list of 4x4"""
    if not HAS_PINOCCHIO:
        return None
    pin.forwardKinematics(model, data, np.asarray(q, dtype=float))
    return [data.oMi[i].homogeneous.copy() for i in range(model.njoints)]


# ═══════════════════════════════════════════════════════
#  ch05: 속도 기구학 — Jacobian
# ═══════════════════════════════════════════════════════

def pin_jacobian(model, data, q, frame_name=None, rf="world"):
    """Pinocchio Jacobian (6 x nv)

    rf: "world" → WORLD, "local" → LOCAL, "aligned" → LOCAL_WORLD_ALIGNED
    """
    if not HAS_PINOCCHIO:
        return None
    q = np.asarray(q, dtype=float)
    rf_map = {
        "world": pin.WORLD,
        "local": pin.LOCAL,
        "aligned": pin.LOCAL_WORLD_ALIGNED,
    }
    reference_frame = rf_map.get(rf, pin.LOCAL_WORLD_ALIGNED)

    if frame_name:
        pin.computeJointJacobians(model, data, q)
        pin.updateFramePlacements(model, data)
        fid = model.getFrameId(frame_name)
        return pin.getFrameJacobian(model, data, fid, reference_frame).copy()
    else:
        pin.computeJointJacobians(model, data, q)
        jid = model.njoints - 1
        return pin.getJointJacobian(model, data, jid, reference_frame).copy()


def pin_jacobian_dt(model, data, q, dq, frame_name=None, rf="aligned"):
    """Jacobian 시간 미분 dJ/dt (6 x nv)"""
    if not HAS_PINOCCHIO:
        return None
    q = np.asarray(q, dtype=float)
    dq = np.asarray(dq, dtype=float)
    rf_map = {
        "world": pin.WORLD,
        "local": pin.LOCAL,
        "aligned": pin.LOCAL_WORLD_ALIGNED,
    }
    reference_frame = rf_map.get(rf, pin.LOCAL_WORLD_ALIGNED)

    pin.computeJointJacobiansTimeVariation(model, data, q, dq)
    if frame_name:
        pin.updateFramePlacements(model, data)
        fid = model.getFrameId(frame_name)
        return pin.getFrameJacobianTimeVariation(
            model, data, fid, reference_frame
        ).copy()
    else:
        jid = model.njoints - 1
        return pin.getJointJacobianTimeVariation(
            model, data, jid, reference_frame
        ).copy()


# ═══════════════════════════════════════════════════════
#  ch06: 역기구학 — IK 헬퍼
# ═══════════════════════════════════════════════════════

def pin_ik(model, data, target_T, frame_name=None, q0=None,
           eps=1e-4, max_iter=1000, dt=0.1, damp=1e-12):
    """Pinocchio 기반 IK (damped least squares)

    target_T: 4x4 목표 변환 행렬
    Returns: (q_sol, success, final_err)
    """
    if not HAS_PINOCCHIO:
        return None, False, None

    oMdes = pin.SE3(np.asarray(target_T, dtype=float))
    q = np.asarray(q0, dtype=float) if q0 is not None else pin.neutral(model)

    if frame_name:
        fid = model.getFrameId(frame_name)
        use_frame = True
    else:
        jid = model.njoints - 1
        use_frame = False

    for i in range(max_iter):
        pin.forwardKinematics(model, data, q)
        if use_frame:
            pin.updateFramePlacements(model, data)
            oMcur = data.oMf[fid]
        else:
            oMcur = data.oMi[jid]

        iMd = oMcur.actInv(oMdes)
        err = pin.log(iMd).vector
        if np.linalg.norm(err) < eps:
            return q.copy(), True, np.linalg.norm(err)

        if use_frame:
            J = pin.computeFrameJacobian(model, data, q, fid)
        else:
            J = pin.computeJointJacobian(model, data, q, jid)

        J = -np.dot(pin.Jlog6(iMd.inverse()), J)
        v = -J.T @ np.linalg.solve(J @ J.T + damp * np.eye(6), err)
        q = pin.integrate(model, q, v * dt)

    return q.copy(), False, np.linalg.norm(err)


# ═══════════════════════════════════════════════════════
#  ch08: 동역학 — RNEA, CRBA, ABA, Energy
# ═══════════════════════════════════════════════════════

def pin_rnea(model, data, q, dq, ddq):
    """역동역학 (RNEA): τ = M·ddq + C·dq + g"""
    if not HAS_PINOCCHIO:
        return None
    return pin.rnea(
        model, data,
        np.asarray(q, dtype=float),
        np.asarray(dq, dtype=float),
        np.asarray(ddq, dtype=float),
    ).copy()


def pin_mass_matrix(model, data, q):
    """관성 행렬 M(q) (CRBA)"""
    if not HAS_PINOCCHIO:
        return None
    return pin.crba(model, data, np.asarray(q, dtype=float)).copy()


def pin_gravity(model, data, q):
    """중력 벡터 g(q)"""
    if not HAS_PINOCCHIO:
        return None
    return pin.computeGeneralizedGravity(
        model, data, np.asarray(q, dtype=float)
    ).copy()


def pin_coriolis(model, data, q, dq):
    """코리올리 행렬 C(q, dq)"""
    if not HAS_PINOCCHIO:
        return None
    return pin.computeCoriolisMatrix(
        model, data,
        np.asarray(q, dtype=float),
        np.asarray(dq, dtype=float),
    ).copy()


def pin_nle(model, data, q, dq):
    """비선형 효과 h(q,dq) = C(q,dq)·dq + g(q)"""
    if not HAS_PINOCCHIO:
        return None
    return pin.nonLinearEffects(
        model, data,
        np.asarray(q, dtype=float),
        np.asarray(dq, dtype=float),
    ).copy()


def pin_aba(model, data, q, dq, tau):
    """순동역학 (ABA): ddq = M^{-1}(τ - h)"""
    if not HAS_PINOCCHIO:
        return None
    return pin.aba(
        model, data,
        np.asarray(q, dtype=float),
        np.asarray(dq, dtype=float),
        np.asarray(tau, dtype=float),
    ).copy()


def pin_minverse(model, data, q):
    """M^{-1}(q) — ABA 기반 역관성 행렬"""
    if not HAS_PINOCCHIO:
        return None
    return pin.computeMinverse(
        model, data, np.asarray(q, dtype=float)
    ).copy()


def pin_kinetic_energy(model, data, q, dq):
    """운동 에너지 K = 0.5 * dq^T M dq"""
    if not HAS_PINOCCHIO:
        return None
    return pin.computeKineticEnergy(
        model, data,
        np.asarray(q, dtype=float),
        np.asarray(dq, dtype=float),
    )


def pin_potential_energy(model, data, q):
    """위치 에너지"""
    if not HAS_PINOCCHIO:
        return None
    return pin.computePotentialEnergy(
        model, data, np.asarray(q, dtype=float),
    )


# ═══════════════════════════════════════════════════════
#  ch08 추가: 동역학 미분 (RNEA derivatives)
# ═══════════════════════════════════════════════════════

def pin_rnea_derivatives(model, data, q, dq, ddq):
    """RNEA 편미분 → (dtau_dq, dtau_dv, dtau_da=M)"""
    if not HAS_PINOCCHIO:
        return None, None, None
    return pin.computeRNEADerivatives(
        model, data,
        np.asarray(q, dtype=float),
        np.asarray(dq, dtype=float),
        np.asarray(ddq, dtype=float),
    )


# ═══════════════════════════════════════════════════════
#  ch05/ch08: 질량 중심 (Center of Mass)
# ═══════════════════════════════════════════════════════

def pin_com(model, data, q):
    """질량 중심 위치 (3-vector)"""
    if not HAS_PINOCCHIO:
        return None
    return pin.centerOfMass(
        model, data, np.asarray(q, dtype=float)
    ).copy()


def pin_com_jacobian(model, data, q):
    """질량 중심 야코비안 (3 x nv)"""
    if not HAS_PINOCCHIO:
        return None
    return pin.jacobianCenterOfMass(
        model, data, np.asarray(q, dtype=float)
    ).copy()


def pin_total_mass(model):
    """전체 질량"""
    if not HAS_PINOCCHIO:
        return None
    return pin.computeTotalMass(model)


# ═══════════════════════════════════════════════════════
#  ch11: 제어 — regressor, static torque
# ═══════════════════════════════════════════════════════

def pin_joint_torque_regressor(model, data, q, dq, ddq):
    """관절 토크 리그레서 Y: τ = Y(q,dq,ddq) · π"""
    if not HAS_PINOCCHIO:
        return None
    return pin.computeJointTorqueRegressor(
        model, data,
        np.asarray(q, dtype=float),
        np.asarray(dq, dtype=float),
        np.asarray(ddq, dtype=float),
    ).copy()


def pin_static_torque(model, data, q):
    """정적 토크 (중력 + 외력 없음) = g(q)와 동일"""
    if not HAS_PINOCCHIO:
        return None
    # fext = 0이면 computeStaticTorque = gravity
    return pin.computeGeneralizedGravity(
        model, data, np.asarray(q, dtype=float)
    ).copy()


# ═══════════════════════════════════════════════════════
#  유틸리티: 설정, 비교
# ═══════════════════════════════════════════════════════

def pin_random_config(model):
    """관절 제한 내 랜덤 configuration"""
    if not HAS_PINOCCHIO:
        return None
    return pin.randomConfiguration(model)


def pin_neutral(model):
    """중립 configuration"""
    if not HAS_PINOCCHIO:
        return None
    return pin.neutral(model)


def pin_integrate(model, q, dq):
    """configuration 적분: q ⊕ dq"""
    if not HAS_PINOCCHIO:
        return None
    return pin.integrate(
        model,
        np.asarray(q, dtype=float),
        np.asarray(dq, dtype=float),
    )


def pin_difference(model, q1, q2):
    """configuration 차이: q2 ⊖ q1"""
    if not HAS_PINOCCHIO:
        return None
    return pin.difference(
        model,
        np.asarray(q1, dtype=float),
        np.asarray(q2, dtype=float),
    )


def compare(name, my_result, pin_result, tol=1e-4):
    """두 결과 비교 출력"""
    if pin_result is None:
        print(f"[{name}] pinocchio 없음 — 스킵")
        return
    diff = np.linalg.norm(np.asarray(my_result) - np.asarray(pin_result))
    status = "✓ PASS" if diff < tol else "✗ FAIL"
    print(f"[{name}] {status}  diff={diff:.2e}")


# ═══════════════════════════════════════════════════════
#  ch07: 폐연쇄 기구학 — 구속 동역학
# ═══════════════════════════════════════════════════════

def pin_load_with_geometry(urdf_path, package_dirs=None):
    """URDF 로드 → (model, collision_model, visual_model, data, geom_data)

    충돌 검사(ch10)와 구속 동역학(ch07)에 필요.
    hpp-fcl이 없으면 geometry model은 None.
    """
    if not HAS_PINOCCHIO:
        return None, None, None, None, None
    try:
        if package_dirs:
            model, collision_model, visual_model = pin.buildModelsFromUrdf(
                urdf_path, package_dirs=package_dirs
            )
        else:
            model, collision_model, visual_model = pin.buildModelsFromUrdf(
                urdf_path
            )
        data = model.createData()
        geom_data = collision_model.createData()
        return model, collision_model, visual_model, data, geom_data
    except Exception as e:
        print(f"[pin_utils] geometry 로드 실패 (hpp-fcl 필요): {e}")
        model = pin.buildModelFromUrdf(urdf_path)
        data = model.createData()
        return model, None, None, data, None


def pin_constrained_dynamics(model, data, q, dq, tau,
                             contact_models, contact_datas):
    """구속 조건 하 순동역학: ddq 반환

    contact_models: list of pin.RigidConstraintModel
    contact_datas:  list of pin.RigidConstraintData
    """
    if not HAS_PINOCCHIO:
        return None
    return pin.constraintDynamics(
        model, data,
        np.asarray(q, dtype=float),
        np.asarray(dq, dtype=float),
        np.asarray(tau, dtype=float),
        contact_models, contact_datas,
    ).copy()


def pin_forward_dynamics_constrained(model, data, q, dq, tau,
                                     J_constraint, gamma, damping=0.0):
    """구속 자코비안 기반 순동역학 (legacy API)

    J_constraint: 구속 자코비안 (n_c x nv)
    gamma: 구속 드리프트 (n_c)
    """
    if not HAS_PINOCCHIO:
        return None
    return pin.forwardDynamics(
        model, data,
        np.asarray(q, dtype=float),
        np.asarray(dq, dtype=float),
        np.asarray(tau, dtype=float),
        np.asarray(J_constraint, dtype=float),
        np.asarray(gamma, dtype=float),
        damping,
    ).copy()


def pin_impulse_dynamics(model, data, q, v_before, J_constraint,
                         r_coeff=0.0, damping=0.0):
    """충격 동역학: 접촉 순간 속도 점프 계산

    v_before: 충돌 전 속도
    r_coeff: 반발 계수 (0=완전 비탄성, 1=완전 탄성)
    """
    if not HAS_PINOCCHIO:
        return None
    return pin.impulseDynamics(
        model, data,
        np.asarray(q, dtype=float),
        np.asarray(v_before, dtype=float),
        np.asarray(J_constraint, dtype=float),
        r_coeff, damping,
    ).copy()


def pin_constraint_jacobian(model, data, contact_models, contact_datas):
    """구속 자코비안 행렬 반환"""
    if not HAS_PINOCCHIO:
        return None
    return pin.getConstraintsJacobian(
        model, data, contact_models, contact_datas
    ).copy()


# ═══════════════════════════════════════════════════════
#  ch10: 동작 계획 — 충돌 검사, 도달 가능 공간
# ═══════════════════════════════════════════════════════

def pin_compute_collisions(model, data, geom_model, geom_data, q,
                           stop_at_first=True):
    """configuration q에서 충돌 여부 검사

    Returns: True if any collision detected
    """
    if not HAS_PINOCCHIO or geom_model is None:
        return None
    return pin.computeCollisions(
        model, data, geom_model, geom_data,
        np.asarray(q, dtype=float),
        stop_at_first,
    )


def pin_compute_distances(model, data, geom_model, geom_data, q):
    """configuration q에서 모든 충돌 쌍의 거리 계산

    Returns: 최소 거리 (geom_data.distanceResults에 상세 결과)
    """
    if not HAS_PINOCCHIO or geom_model is None:
        return None
    pin.computeDistances(
        model, data, geom_model, geom_data,
        np.asarray(q, dtype=float),
    )
    # 모든 쌍의 최소 거리 반환
    min_dist = float('inf')
    for dr in geom_data.distanceResults:
        if dr.min_distance < min_dist:
            min_dist = dr.min_distance
    return min_dist


def pin_collision_check_path(model, data, geom_model, geom_data,
                             q_start, q_end, n_steps=10):
    """경로 상의 충돌 검사 (선형 보간)

    Returns: (collision_free: bool, first_collision_t: float or None)
    """
    if not HAS_PINOCCHIO or geom_model is None:
        return None, None
    q_s = np.asarray(q_start, dtype=float)
    q_e = np.asarray(q_end, dtype=float)
    for i in range(n_steps + 1):
        t = i / n_steps
        q = q_s + t * (q_e - q_s)
        if pin.computeCollisions(model, data, geom_model, geom_data, q, True):
            return False, t
    return True, None


def pin_reachable_workspace(model, q0, time_horizon, frame_id,
                            n_samples=5, facet_dims=3):
    """도달 가능 작업 공간 점군 계산

    Returns: vertex matrix (N x 3)
    """
    if not HAS_PINOCCHIO:
        return None
    try:
        return pin.reachableWorkspace(
            model, np.asarray(q0, dtype=float),
            time_horizon, frame_id, n_samples, facet_dims,
        )
    except AttributeError:
        print("[pin_utils] reachableWorkspace 미지원 (빌드 옵션 확인)")
        return None


def pin_reachable_workspace_hull(model, q0, time_horizon, frame_id,
                                 n_samples=5, facet_dims=3):
    """도달 가능 작업 공간 convex hull

    Returns: (vertex, faces)
    """
    if not HAS_PINOCCHIO:
        return None, None
    try:
        return pin.reachableWorkspaceHull(
            model, np.asarray(q0, dtype=float),
            time_horizon, frame_id, n_samples, facet_dims,
        )
    except AttributeError:
        print("[pin_utils] reachableWorkspaceHull 미지원 (빌드 옵션 확인)")
        return None, None


def pin_update_geometry(model, data, geom_model, geom_data, q):
    """충돌 객체 위치 업데이트 (FK + geometry placement)"""
    if not HAS_PINOCCHIO or geom_model is None:
        return
    pin.updateGeometryPlacements(
        model, data, geom_model, geom_data,
        np.asarray(q, dtype=float),
    )
