# ModernRobotics

Modern Robotics: Mechanics, Planning, and Control (Kevin Lynch, Frank Park) 교재 기반 실습 코드 저장소.
ch02~ch08 구현을 UR5e Pick & Place 시뮬레이션으로 통합.

## 폴더 구조

```
ch02_configuration_space/       # C-space, 위상, Grübler 공식, 제약조건
ch03_rigid_body_motion/         # SO(3), SE(3), exp/log, Adjoint, 스크류
ch04_forward_kinematics/        # PoE 기반 순운동학 (UR5, UR5e)
ch05_velocity_kinematics/       # 물체/공간 자코비안 (UR5e)
ch06_inverse_kinematics/        # 수치 IK (Newton-Raphson)
ch07_closed_chain_kinematics/   # 폐연쇄 기구학, 파지 해석
ch08_dynamics/                  # RNEA, 질량 행렬, 중력/코리올리 토크 (UR5e)
kinematics_pick_and_place/      # ch02~08 통합 Pick & Place 시뮬레이션
params/                         # UR5e 기구학/동역학 파라미터 공통 모듈
pin_utils/                      # Pinocchio 검증 유틸리티
mujoco_menagerie/               # MuJoCo 로봇 모델 (git clone, .gitignore)
urdf_files_dataset/             # URDF 파일 모음 (git clone, .gitignore)
```

## 챕터별 구현 현황

| 챕터 | 주제 | 검증 |
|------|------|------|
| ch02 | Configuration Space, Grübler, 제약조건 | - |
| ch03 | SO(3)/SE(3), exp/log, Adjoint | ✓ Pinocchio PASS |
| ch04 | PoE 순운동학 | ✓ Pinocchio, MuJoCo PASS |
| ch05 | 물체/공간 자코비안 | ✓ Pinocchio, MuJoCo PASS |
| ch06 | 수치 역기구학 | ✓ Pinocchio, MuJoCo PASS |
| ch07 | 폐연쇄 기구학, 파지 해석 | ✓ Pinocchio, 수학적 검증 PASS |
| ch08 | RNEA, 질량 행렬, 동역학 | ✗ FAIL (분석 중, `ch08_dynamics/FAIL_analysis.md`) |

## 비교 검증 체계

각 챕터에는 직접 구현 결과를 외부 라이브러리와 비교하는 스크립트가 포함되어 있다.

- `compared_mr2pin.py` — Pinocchio (URDF 기반) 비교
- `compared_mr2mujoco.py` — MuJoCo (MJCF 기반) 비교

## Pick & Place 통합 시뮬레이션

`kinematics_pick_and_place/`에서 3가지 IK 방식을 비교한다:

| 방식 | 파일 | 특징 |
|------|------|------|
| MR OSQP | `mr_pick_and_place.py` | ch08 M(θ) 목적함수 + 관절 제한 QP |
| MuJoCo DLS | `mujoco_pick_and_place.py` | mj_jacBody + damped least squares |
| Pinocchio | `pin_pick_and_place.py` | SE(3) 오차 + Jlog6 보정 |

## 대상 로봇

| 로봇 | 용도 | 단위 |
|------|------|------|
| UR5 | ch04~ch05 기구학 | mm |
| UR5e | ch05~ch08, pick & place | m (SI) |
| Cassie | ch07 폐연쇄 (예정) | m |

## 환경 설정

```bash
conda create -n mr python=3.13
conda activate mr
pip install numpy scipy pinocchio mujoco sympy osqp matplotlib
```

외부 데이터 (프로젝트 루트에 위치):

```bash
git clone https://github.com/Daniella1/urdf_files_dataset.git
git clone https://github.com/google-deepmind/mujoco_menagerie.git
```

## 실행 예시

```bash
conda activate mr

# 챕터별 메인
python ch03_rigid_body_motion/modern_robotics_ch03.py
python ch04_forward_kinematics/modern_robotics_ch04.py
python ch05_velocity_kinematics/modern_robotics_ch05.py
python ch08_dynamics/modern_robotics_ch08.py

# 비교 검증
python ch03_rigid_body_motion/compared_mr2pin.py
python ch04_forward_kinematics/compared_mr2pin.py
python ch04_forward_kinematics/compared_mr2mujoco.py
python ch05_velocity_kinematics/compared_mr2pin.py
python ch05_velocity_kinematics/compared_mr2mujoco.py
python ch06_inverse_kinematics/compared_mr2pin.py
python ch06_inverse_kinematics/compared_mr2mujoco.py
python ch07_closed_chain_kinematics/compared_mr2pin.py
python ch07_closed_chain_kinematics/compared_mr2mujoco.py
python ch08_dynamics/compared_mr2pin.py
python ch08_dynamics/compared_mr2mujoco.py

# Pick & Place 시뮬레이션
python kinematics_pick_and_place/mr_pick_and_place.py
python kinematics_pick_and_place/mujoco_pick_and_place.py
python kinematics_pick_and_place/pin_pick_and_place.py