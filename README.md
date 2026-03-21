# ModernRobotics

Modern Robotics: Mechanics, Planning, and Control (Kevin Lynch, Frank Park) 교재 기반 실습 코드 저장소

## 폴더 구조

```
ch02_configuration_space/    # C-space, 위상, Grübler 공식, 제약조건
ch03_rigid_body_motion/      # SO(3), SE(3), exp/log, Adjoint, 스크류
ch04_forward_kinematics/     # PoE 기반 순운동학 (UR5)
ch05_velocity_kinematics/    # 물체/공간 자코비안 (UR5)
ch06_inverse_kinematics/     # 역기구학 (스켈레톤)
ch07_closed_chain_kinematics/# 폐연쇄 기구학, Cassie 로봇 분석
ch08_dynamics/               # 동역학, twist-wrench, RNEA (UR5e)
pin_utils/                   # Pinocchio 검증 유틸리티
mujoco_menagerie/            # MuJoCo 로봇 모델 모음 (git clone)
urdf_files_dataset/          # URDF 파일 모음 (git clone, .gitignore)
```

## 비교 검증 체계

각 챕터에는 직접 구현한 결과를 외부 라이브러리와 비교하는 스크립트가 포함되어 있다.

- `compared_mr2pin.py` — Pinocchio (URDF 기반) 비교
- `compared_mr2mujoco.py` — MuJoCo (MJCF 기반) 비교

## 대상 로봇

| 로봇 | 용도 | 단위 |
|------|------|------|
| UR5 | ch04~ch05 기구학 | mm |
| UR5e | ch08 동역학 | m (SI) |
| Cassie | ch07 폐연쇄 | m |

## 환경 설정

### Conda 가상환경

```bash
conda create -n mr python=3.13
conda activate mr
pip install numpy scipy pinocchio mujoco sympy
```

### 외부 데이터

```bash
git clone https://github.com/Daniella1/urdf_files_dataset.git
git clone https://github.com/google-deepmind/mujoco_menagerie.git
```

두 폴더 모두 프로젝트 루트에 위치시킨다.

## 실행 예시

```bash
conda activate mr

# 각 챕터 메인 실행
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
python ch08_dynamics/compared_mr2pin.py
python ch08_dynamics/compared_mr2mujoco.py

# Cassie 폐연쇄 분석 + 뷰어 스폰
python ch07_closed_chain_kinematics/cassie_test.py
python ch07_closed_chain_kinematics/cassie_test.py --view
```
