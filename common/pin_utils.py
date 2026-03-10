"""
Pinocchio 검증 유틸리티

직접 구현한 결과를 Pinocchio와 비교하는 헬퍼 함수들.
Pinocchio가 설치되어 있지 않으면 graceful하게 스킵.
"""

import numpy as np

try:
    import pinocchio as pin
    HAS_PINOCCHIO = True
except ImportError:
    HAS_PINOCCHIO = False
    print("[pin_utils] pinocchio not found — 검증 함수 비활성화")


def load_urdf(urdf_path):
    """URDF 로드 → (model, data)"""
    if not HAS_PINOCCHIO:
        return None, None
    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()
    return model, data


def pin_fk(model, data, q, frame_name=None):
    """Pinocchio FK → 4x4 변환 행렬"""
    if not HAS_PINOCCHIO:
        return None
    pin.forwardKinematics(model, data, np.asarray(q))
    if frame_name:
        pin.updateFramePlacements(model, data)
        fid = model.getFrameId(frame_name)
        return data.oMf[fid].homogeneous
    else:
        # 마지막 조인트 프레임
        return data.oMi[-1].homogeneous


def pin_jacobian(model, data, q, frame_name=None):
    """Pinocchio Jacobian (world frame)"""
    if not HAS_PINOCCHIO:
        return None
    q = np.asarray(q)
    pin.computeJointJacobians(model, data, q)
    if frame_name:
        pin.updateFramePlacements(model, data)
        fid = model.getFrameId(frame_name)
        return pin.getFrameJacobian(model, data, fid, pin.LOCAL_WORLD_ALIGNED)
    else:
        jid = model.njoints - 1
        return pin.getJointJacobian(model, data, jid, pin.LOCAL_WORLD_ALIGNED)


def pin_rnea(model, data, q, dq, ddq):
    """Pinocchio RNEA → τ"""
    if not HAS_PINOCCHIO:
        return None
    return pin.rnea(model, data,
                    np.asarray(q), np.asarray(dq), np.asarray(ddq))


def pin_mass_matrix(model, data, q):
    """Pinocchio CRBA → M(q)"""
    if not HAS_PINOCCHIO:
        return None
    return pin.crba(model, data, np.asarray(q))


def pin_gravity(model, data, q):
    """Pinocchio gravity vector → g(q)"""
    if not HAS_PINOCCHIO:
        return None
    return pin.computeGeneralizedGravity(model, data, np.asarray(q))


def compare(name, my_result, pin_result, tol=1e-4):
    """두 결과 비교 출력"""
    if pin_result is None:
        print(f"[{name}] pinocchio 없음 — 스킵")
        return
    diff = np.linalg.norm(np.asarray(my_result) - np.asarray(pin_result))
    status = "✓ PASS" if diff < tol else "✗ FAIL"
    print(f"[{name}] {status}  diff={diff:.2e}")
