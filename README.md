# Modern Robotics — Python 구현

Kevin Lynch "Modern Robotics" 교재 기반 알고리즘 구현 + Pinocchio 검증

## 폴더 구조

```
modern_robotics/
├── common/                          ← 공유 로봇 정의 + Pinocchio 검증 유틸
│   ├── robot_def.py                 ← 3DOF RRR 로봇 (DH + PoE 파라미터)
│   └── pin_utils.py                 ← Pinocchio 비교 헬퍼
│
├── ch02_configuration_space/        ← 자유도, C-space 시각화
├── ch03_rigid_body_motion/          ← SO(3), SE(3), 오일러각, 쿼터니언
├── ch04_forward_kinematics/         ← PoE FK, DH FK, URDF→PoE 추출
├── ch05_velocity_kinematics/        ← 자코비안, 매니퓰러빌리티, null-space
├── ch06_inverse_kinematics/         ← Newton IK, DLS, 해석적 IK, 관절제한
├── ch07_kinematics_of_closed_chains/← 병렬 메커니즘, 4절 링크
├── ch08_dynamics/                   ← 라그랑지안, RNEA, 모터/기어링, 에너지
├── ch09_trajectory_generation/      ← 다항식 궤적, 스크류 보간, 시간 최적
├── ch10_motion_planning/            ← RRT, RRT*, PRM, A*, 포텐셜 필드
├── ch11_robot_control/              ← PD, CTC, 임피던스, 적응, 작업공간 제어
├── ch12_grasping/                   ← 접촉, 마찰, force/form closure, 품질지표
├── ch13_wheeled_mobile_robots/      ← 차동구동, 비홀로노믹, 이동매니퓰레이터
│
├── robot.py                         ← 기존 3DOF 시각화 (standalone)
└── FK.py                            ← 기존 DH FK (standalone)
```

각 챕터에 `exercises/` 폴더가 있음 — 교재 연습 문제를 코드로 풀어보는 공간

## 추천 순서

1. ch03 (강체 운동) — 회전, 변환, 오일러각, 쿼터니언
2. ch04 (정기구학) — PoE FK
3. ch05 (속도 기구학) — 자코비안, 매니퓰러빌리티, null-space
4. ch06 (역기구학) — 수치 IK, 해석적 IK, 관절 제한
5. ch08 (동역학) — M, C, g, RNEA, 모터/기어링
6. ch09 (궤적 생성) — 궤적 보간, 시간 최적
7. ch11 (제어) — PD, CTC, 임피던스, 적응 제어
8. ch10, ch12, ch13 — 응용

## 검증 3단계

1. 직접 구현 → 수식과 일치하는지 단위 테스트
2. Pinocchio 결과와 수치 비교 (`common/pin_utils.py`)
3. MuJoCo 물리 시뮬에서 실제 동작 확인 (ch11)
