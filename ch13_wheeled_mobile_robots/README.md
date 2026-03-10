# Chapter 13: 차륜 이동 로봇 (Wheeled Mobile Robots)

## 구현 목록

### `diff_drive.py` — 차동 구동
- `diff_drive_fk(v, omega, dt)`: (v, ω) → (x, y, θ) 적분
- `diff_drive_ik(dx, dy, dtheta)`: 원하는 이동 → (v_L, v_R)
- 오도메트리 시뮬레이션

### `odometry.py` — 오도메트리 (추가)
- `wheel_odometry(encoder_L, encoder_R, params)`: 엔코더 → (x, y, θ)
- `odometry_error_propagation(sigma_enc, params)`: 오차 전파 분석
- 누적 오차 시뮬레이션 (노이즈 모델 포함)

### `nonholonomic.py` — 비홀로노믹 구속 (추가)
- `pfaffian_constraint(q)`: A(q)·dq = 0 형태의 구속 행렬
- `is_controllable(A)`: Lie bracket 기반 제어 가능성 판별
- diff-drive의 비홀로노믹 구속 수식 유도 및 검증

### `omnidirectional.py` — 전방향 이동
- 메카넘 휠 기구학
- H 행렬 기반 바퀴 속도 ↔ body twist 변환

### `trajectory_tracking.py` — 궤적 추종
- 피드백 선형화 기반 추종 제어
- 원형/8자 궤적 추종 시뮬레이션

### `mobile_manipulator.py` — 이동 매니퓰레이터 (추가)
- `mobile_manipulator_jacobian(J_base, J_arm)`: 통합 자코비안
- `whole_body_ik(robot, T_target, q_base, q_arm)`: 베이스 + 팔 통합 IK
- FrJoCo_C의 Husky + UR5e 구성과 직접 대응

### `mobile_viz.py` — 시각화
- 2D 평면 로봇 궤적 애니메이션
- 오도메트리 vs 실제 경로 비교

### FrJoCo_C 연계
- Husky diff-drive 파라미터로 검증 가능

## 시각화

### ★ 가치 높음
- **궤적 추종 애니메이션**
  - 2D 평면에서 로봇이 목표 궤적을 따라가는 애니메이션
  - 목표 경로 (점선) vs 실제 경로 (실선) 비교
  - 원형, 8자, S자 궤적 추종
- **오도메트리 드리프트**
  - 이상적 오도메트리 vs 노이즈 추가 오도메트리 경로 비교
  - 시간이 지남에 따라 누적 오차가 커지는 것 시각화

### ☆ 가치 보통
- **메카넘 휠 전방향 이동**
  - 4개 메카넘 휠의 속도 벡터 → body twist 변환 시각화
  - 횡이동, 대각선 이동, 제자리 회전 등 다양한 모션 데모
- **비홀로노믹 구속 시각화** (추가)
  - diff-drive가 옆으로 못 가는 것을 C-space (x, y, θ)에서 허용 속도 방향으로 표시
  - 같은 (x, y)에 도달하는 여러 경로 비교 (평행 주차 등)
- **이동 매니퓰레이터 workspace** (추가)
  - 베이스 고정 시 arm workspace vs 베이스 이동 시 확장된 workspace 비교
