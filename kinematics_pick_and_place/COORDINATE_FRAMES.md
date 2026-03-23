# 좌표계 불일치 문제: URDF vs MJCF 파라미터 추출

UR5e Pick & Place 구현 과정에서 발견한 URDF 기반 파라미터와 MuJoCo MJCF 기반 파라미터의
좌표 규약 차이를 정리한다.

## 문제 요약

같은 관절각 θ에 대해 **URDF 기준 MR FK**와 **MuJoCo FK**가 전혀 다른 EE 위치를 반환한다.

```
q_home = [-π/2, -π/2, π/2, -π/2, -π/2, 0]

MR FK (URDF params):  [-0.1333,  0.3922, -0.2624]
MuJoCo FK:            [-0.134,   0.492,   0.448]
```

단순 Rz(π) 변환으로는 매핑 불가능하다. x, y, z 모두 다르다.

## 원인: URDF → MJCF 변환 시 좌표 규약 차이

### 1. Base 프레임 회전

MuJoCo menagerie의 UR5e MJCF에는 base body에 `quat="0 0 0 -1"`이 적용되어 있다.
이는 z축 180° 회전(Rz(π))으로, URDF→MJCF 변환 시 관례 차이를 보정하기 위한 것이다.

```xml
<!-- ur5e_gripper.xml -->
<body name="base" quat="0 0 0 -1" childclass="ur5e">
```

### 2. 조인트 축 방향 및 링크 프레임

URDF와 MJCF는 조인트 축 방향과 링크 프레임 정의가 다르다.
URDF의 `<origin rpy="...">` 태그로 표현되는 부모-자식 프레임 간 회전이
MJCF에서는 body의 `quat` 속성과 joint의 `axis` 속성으로 다르게 인코딩된다.

### 3. EE 프레임 정의

| 소스 | EE 프레임 | zero config 위치 |
|------|-----------|-----------------|
| URDF (MR `M_e`) | tool0 (wrist_3 끝) | `[-0.817, -0.191, 0.006]` |
| MJCF attachment_site | wrist_3 site | `[-0.817, -0.234, 0.063]` |
| MJCF gripper_palm | 그리퍼 손바닥 | `[-0.817, -0.274, 0.063]` |

URDF의 tool0과 MJCF의 attachment_site도 위치가 다르고,
gripper_palm은 그리퍼 오프셋이 추가로 포함된다.

## 두 가지 파라미터 추출 방식

### 방식 A: URDF에서 추출 (params/ur5e.py)

DH 파라미터 또는 URDF의 링크 길이/오프셋으로부터 직접 계산한다.

```python
# 영 위치 EE 변환행렬
R_e = np.array([[-1, 0,  0],
                [ 0, 0,  1],
                [ 0, 1,  0]])
p_e = np.array([[-(L1 + L2)],
                [-(W1 - H2 + W2)],
                [H1]])
M_e = np.block([[R_e, p_e], [0, 0, 0, 1]])

# 공간꼴 스크류 축: 조인트 위치 q_i와 축 방향 w_i로 계산
v_i = -skew(w_i) @ q_i
S_i = [w_i; v_i]
```

이 파라미터는 **URDF 좌표 규약**에서만 유효하다.
Pinocchio, MR 교재 예제 등 URDF를 직접 파싱하는 라이브러리와 호환된다.

### 방식 B: MuJoCo에서 추출 (mr_pick_and_place.py)

MuJoCo 모델을 로드하고, zero config에서 수치적으로 M과 스크류 축을 추출한다.

```python
def extract_mr_params_from_mujoco(scene_xml, ee_body_name):
    model = mujoco.MjModel.from_xml_path(scene_xml)
    data = mujoco.MjData(model)
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_body_name)

    # zero config에서 M 추출
    for i in range(6): data.qpos[i] = 0.0
    mujoco.mj_forward(model, data)
    M[:3,:3] = data.xmat[ee_id].reshape(3,3)
    M[:3,3]  = data.xpos[ee_id]

    # 각 조인트의 공간꼴 스크류 축 추출
    for i in range(6):
        w = body_rot @ joint_axis   # world frame 조인트 축
        q = body_pos + body_rot @ joint_pos  # world frame 조인트 위치
        v = -cross(w, q)
        S_i = [w; v]
```

이 파라미터는 **MuJoCo world 좌표계**에서 유효하다.
MR FK(PoE)로 계산한 결과가 MuJoCo FK와 정확히 일치한다.

```
MR FK (MuJoCo params): [-0.134,  0.492,  0.448]
MuJoCo FK:             [-0.134,  0.492,  0.448]
Match: True
```

## MR vs Pinocchio: IK 엔진 차이

### Modern Robotics (MR)

- Product of Exponentials (PoE) 공식으로 FK/자코비안 계산
- `M_e`, `Blist`(또는 `Slist`)를 직접 제공해야 함
- 파라미터 소스에 따라 좌표계가 결정됨
  - URDF에서 추출 → URDF 좌표계
  - MuJoCo에서 추출 → MuJoCo world 좌표계
- IK는 자코비안 기반 반복법 (이 프로젝트에서는 OSQP QP solver 사용)

```python
# MR IK 호출
T_sd = Rp2Trans(R_target, p_target)  # 목표 pose
q_sol, ok = osqp_ik(Blist_vec, M, T_sd, q_init)
```

### Pinocchio

- URDF 파일을 직접 파싱하여 내부적으로 kinematic tree 구성
- `buildModelFromUrdf(urdf_path)`로 모델 생성, 별도 파라미터 제공 불필요
- **항상 URDF 좌표 규약**을 따름
- SE(3) 오차 + Jlog6 보정을 사용한 damped least squares IK

```python
# Pinocchio IK 호출
model = pin.buildModelFromUrdf(urdf_path)
oMdes = pin.SE3(R_target, p_target)  # 목표 pose (URDF 좌표계)
# ... 반복: err = log(oMcur.actInv(oMdes)), J = getFrameJacobian(...)
```

### 비교 정리

| 항목 | MR (PoE) | Pinocchio |
|------|----------|-----------|
| 모델 입력 | M, Blist/Slist (수동 제공) | URDF 파일 경로 |
| 좌표계 | 파라미터 소스에 의존 | 항상 URDF 기준 |
| FK 방식 | `M * exp(B1*θ1) * ... * exp(Bn*θn)` | 내부 kinematic tree 순회 |
| 자코비안 | `BodyJacobian(Blist, θ)` | `getFrameJacobian(model, data, frame_id)` |
| MuJoCo 호환 | MuJoCo에서 추출하면 완벽 호환 | URDF 기준이라 좌표 변환 필요 |
| 그리퍼 오프셋 | EE body를 자유롭게 선택 가능 | URDF에 정의된 프레임만 사용 |

## 실용적 가이드라인

### MuJoCo 시뮬레이션과 함께 쓸 때

1. **MR**: MuJoCo scene에서 M/스크류 축을 추출하면 좌표 변환 없이 동작한다.
   EE body도 gripper_palm 등 원하는 body를 지정할 수 있다.

2. **Pinocchio**: URDF 좌표계 기준이므로 MuJoCo world와의 변환이 필요하다.
   base의 `quat="0 0 0 -1"` 등을 고려한 `mj2pin` / `pin2mj` 변환을 구현해야 한다.
   단, 이 변환은 단순 Rz(π)가 아닐 수 있으므로 주의가 필요하다.

3. **MuJoCo 자체 IK**: `mj_jacBody`로 자코비안을 구해 MuJoCo world 좌표계에서
   직접 IK를 풀면 좌표 변환이 전혀 필요 없다. 가장 간단하지만 MR/Pinocchio의
   알고리즘을 활용할 수 없다.

### 검증 방법

어떤 방식을 쓰든, home config에서 FK 결과가 MuJoCo와 일치하는지 반드시 확인한다:

```python
p_mr, R_mr = mr_fk(q_home)      # MR FK
p_mj = sim.get_ee_pos()          # MuJoCo FK
assert np.allclose(p_mr, p_mj, atol=1e-3), "FK 불일치!"
```
