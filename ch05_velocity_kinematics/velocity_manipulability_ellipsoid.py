# -*- coding: utf-8 -*-
"""
ch05 속도 조작성 타원체 (Velocity Manipulability Ellipsoid) 시각화
MuJoCo viewer에서 UR5e EE 위치에 타원체를 실시간 렌더링

conda env: mr
python ch05_velocity_kinematics/velocity_manipulability_ellipsoid.py
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import mujoco
import mujoco.viewer

np.set_printoptions(precision=4, suppress=True)

# ── MuJoCo 모델 로드 ──
SCENE_XML = os.path.join(os.path.dirname(__file__), '..',
                         'mujoco_menagerie/universal_robots_ur5e/scene.xml')

model = mujoco.MjModel.from_xml_path(SCENE_XML)
data = mujoco.MjData(model)

site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'attachment_site')

# 초기 자세 (home keyframe)
data.qpos[:6] = [-np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0]
mujoco.mj_forward(model, data)

# 타원체 스케일 (시각화용, 너무 크면 줄이기)
ELLIPSOID_SCALE = 0.15


def get_linear_jacobian(model, data):
    """EE site의 선속도 자코비안 Jv (3x6) 추출"""
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    return jacp[:, :6]


def ellipsoid_from_jacobian(Jv):
    """
    선속도 자코비안 Jv (3×n)로부터 조작성 타원체 파라미터 추출.

    A = Jv @ Jv^T  (3×3)
    SVD → 특이값 σ_i = 타원체 반축 길이, 특이벡터 = 타원체 축 방향

    Returns:
        axes_lengths: 특이값 (3,) - 타원체 반축 길이
        rotation: 회전행렬 (3×3) - 타원체 축 방향
    """
    U, sigma, _ = np.linalg.svd(Jv)
    # sigma는 Jv의 특이값, 타원체 반축 = sigma
    return sigma, U


def add_ellipsoid_geom(viewer, pos, rotation, axes, rgba):
    """
    viewer.user_scn에 타원체 geom 추가.
    MuJoCo의 mjGEOM_ELLIPSOID 사용.
    """
    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
        return

    g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    mujoco.mjv_initGeom(
        g,
        type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
        size=axes * ELLIPSOID_SCALE,
        pos=pos,
        mat=rotation.flatten(),
        rgba=rgba,
    )
    g.category = mujoco.mjtCatBit.mjCAT_DECOR
    viewer.user_scn.ngeom += 1


def add_principal_axes(viewer, pos, rotation, axes):
    """타원체 주축을 화살표로 시각화"""
    colors = [
        np.array([1.0, 0.2, 0.2, 0.9]),  # X축 - 빨강
        np.array([0.2, 1.0, 0.2, 0.9]),  # Y축 - 초록
        np.array([0.2, 0.2, 1.0, 0.9]),  # Z축 - 파랑
    ]
    for i in range(3):
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
            return
        direction = rotation[:, i] * axes[i] * ELLIPSOID_SCALE
        end = pos + direction

        g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mujoco.mjv_initGeom(
            g,
            type=mujoco.mjtGeom.mjGEOM_ARROW,
            size=np.array([0.005, 0.005, 0.005]),
            pos=np.zeros(3),
            mat=np.eye(3).flatten(),
            rgba=colors[i],
        )
        mujoco.mjv_connector(
            g,
            mujoco.mjtGeom.mjGEOM_ARROW,
            0.005,
            pos.astype(np.float64),
            end.astype(np.float64),
        )
        g.category = mujoco.mjtCatBit.mjCAT_DECOR
        viewer.user_scn.ngeom += 1


def update_ellipsoid(viewer):
    """매 프레임 호출 - 타원체 업데이트"""
    viewer.user_scn.ngeom = 0  # 이전 프레임 geom 초기화

    # EE 위치 (mj_forward는 메인 루프에서 이미 호출됨)
    ee_pos = data.site_xpos[site_id].copy()

    # 선속도 자코비안 → 타원체 파라미터
    Jv = get_linear_jacobian(model, data)
    sigma, U = ellipsoid_from_jacobian(Jv)

    # 타원체 (반투명 파랑)
    add_ellipsoid_geom(
        viewer, ee_pos, U,
        axes=sigma,
        rgba=np.array([0.2, 0.5, 1.0, 0.3]),
    )

    # 주축 화살표
    add_principal_axes(viewer, ee_pos, U, sigma)


# ── Viewer 실행 ──
print("=" * 50)
print("  Velocity Manipulability Ellipsoid - UR5e")
print("  Control 패널에서 관절을 조작하면 타원체가 실시간 업데이트됩니다")
print("=" * 50)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # ctrl → qpos 직접 반영 (순수 기구학 모드)
        data.qpos[:6] = data.ctrl[:6]
        mujoco.mj_forward(model, data)

        update_ellipsoid(viewer)
        viewer.sync()
