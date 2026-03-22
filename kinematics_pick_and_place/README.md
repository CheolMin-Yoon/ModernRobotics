# Kinematics Pick & Place

ch02~ch08 구현을 통합한 UR5e + 3-finger gripper Pick & Place 시뮬레이션.
IK 방식 3가지(MR OSQP, MuJoCo DLS, Pinocchio)를 비교하고, 실시간 접촉력 및 그래스프 분석을 시각화한다.

## 구성

| 파일 | 설명 |
|------|------|
| `config.py` | 공통 설정 (경로, 관절 이름, IK 파라미터, 그리퍼 자세, 시뮬 파라미터) |
| `osqp_ik.py` | OSQP 기반 라그랑주 IK (ch08 질량 행렬 + 관절 제한) |
| `mr_pick_and_place.py` | MR OSQP IK + MuJoCo 시뮬 |
| `mujoco_pick_and_place.py` | MuJoCo DLS IK + MuJoCo 시뮬 |
| `pin_pick_and_place.py` | Pinocchio IK + MuJoCo 시뮬 |
| `grasp_analysis.py` | 그래스프 행렬, force closure, 폐연쇄 DOF 분석 |

## IK 방식 비교

| 방식 | 파일 | 특징 |
|------|------|------|
| MR OSQP | `mr_pick_and_place.py` | ch08 M(θ) 목적함수, 관절 위치/속도 제한 QP |
| MuJoCo DLS | `mujoco_pick_and_place.py` | mj_jacBody + damped least squares |
| Pinocchio | `pin_pick_and_place.py` | SE(3) 오차 + Jlog6 보정 |

## OSQP IK 수식

매 반복마다 QP를 풀어 관절 변위 dθ를 구한다:

```
min   0.5 * dθᵀ M(θ) dθ
s.t.  J_b dθ = V_b                        (기구학 등식)
      q_lower - θ ≤ dθ ≤ q_upper - θ     (위치 제한)
      -dq_max·dt  ≤ dθ ≤  dq_max·dt      (속도 제한)
```

- `P = M(θ)`: ch08 질량 행렬 (운동에너지 최소화)
- `A = [J_b; I]`: 등식 + 부등식 구속 결합
- OSQP 수렴 실패 시 pseudo-inverse fallback

## 그래스프 분석 (`grasp_analysis.py`)

ch05 자코비안 + ch07 폐연쇄 기구학 이론 적용:

- 그래스프 행렬 G (6×3k): 접촉력 → 물체 wrench
  ```
  G = [ I_3  I_3  ... ]
      [r1×  r2×  ... ]
  ```
- Force closure: rank(G) = 6 + 최소 특이값 > 임계값
- 그래스프 품질: isotropy index = σ_min / σ_max
- 폐연쇄 DOF: Grübler 공식 (그리퍼 12 DOF + 물체 6 DOF - 접촉 구속 3k)

## 웨이포인트 시퀀스

```
home → approach → pre-grasp → grasp → (close gripper)
     → lift → place_above → place → (open gripper) → retreat → home
```

## 실행

```bash
conda activate mr

# MR OSQP IK (ch02~08 통합)
python kinematics_pick_and_place/mr_pick_and_place.py

# MuJoCo DLS IK
python kinematics_pick_and_place/mujoco_pick_and_place.py

# Pinocchio IK
python kinematics_pick_and_place/pin_pick_and_place.py
```

## 의존성

- `ch03_rigid_body_motion`, `ch04_forward_kinematics`, `ch05_velocity_kinematics`, `ch08_dynamics`
- `params/ur5e.py` (UR5e 파라미터)
- `mujoco_gripper/scene.xml` (UR5e + 3-finger gripper MuJoCo scene)
- `osqp`, `mujoco`, `pinocchio`, `matplotlib`
