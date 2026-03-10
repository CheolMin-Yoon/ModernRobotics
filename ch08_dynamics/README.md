# Chapter 8: 개연쇄의 동역학 (Open-Chain Dynamics)

## 구현 목록

### `lagrangian.py` — 라그랑지안 동역학
- `mass_matrix(robot, q)`: M(q) 관성 행렬 직접 계산
- `coriolis(robot, q, dq)`: C(q,dq) 코리올리/원심력 행렬
- `gravity_vector(robot, q)`: g(q) 중력 벡터
- `forward_dynamics(M, C, g, tau)`: ddq = M^{-1}(τ - C·dq - g)
- `inverse_dynamics(M, C, g, ddq)`: τ = M·ddq + C·dq + g

### `rne.py` — Recursive Newton-Euler 알고리즘
- `rne(robot, q, dq, ddq)`: 재귀적 역동역학
  - Forward pass: 속도/가속도 전파
  - Backward pass: 힘/토크 역전파
- 라그랑지안 결과와 교차 검증

### `actuator_dynamics.py` — 모터/기어링 모델 (추가)
- `reflected_inertia(gear_ratio, motor_inertia)`: 기어비에 의한 반사 관성
- `effective_mass_matrix(M, B)`: M_eff = M + B (B = 반사 관성 대각 행렬)
- `friction_model(dq, fc, fv)`: 마찰 토크 = fc·sign(dq) + fv·dq
  - fc: 쿨롱 마찰 계수
  - fv: 점성 마찰 계수
- `motor_torque_to_joint_torque(tau_motor, gear_ratio, efficiency)`: 모터 → 관절 토크 변환
- 기어비가 M(q)에 미치는 영향 분석

### `energy.py` — 에너지 분석 (추가)
- `kinetic_energy(M, dq)`: K = 0.5 · dq^T · M(q) · dq
- `potential_energy(robot, q)`: P = Σ m_i · g^T · p_com_i(q)
- `total_energy(robot, q, dq)`: E = K + P
- 시뮬레이션 중 에너지 보존 확인 (보존계에서 E = const)
- 오일러 vs RK4 적분기의 에너지 드리프트 비교

### `dynamics_sim.py` — 동역학 시뮬레이션
- `simulate(robot, q0, dq0, tau_func, dt, T)`: 오일러/RK4 적분
- 자유 낙하 (τ=0) 시뮬레이션
- 중력 보상 (τ=g(q)) 시뮬레이션

### `mass_properties.py` — 질량 특성
- URDF에서 질량/관성 추출
- 직접 정의한 값과 비교

### Pinocchio 검증
- `pin.crba(model, data, q)` → M(q) 비교
- `pin.rnea(model, data, q, dq, ddq)` → τ 비교
- `pin.computeGeneralizedGravity()` → g(q) 비교

## 시각화

### ★ 가치 높음
- **자유 낙하 시뮬레이션 애니메이션**
  - τ = 0에서 로봇이 중력에 의해 떨어지는 과정
  - q(t), dq(t), 에너지(t) 시계열 플롯 동시 표시
  - 에너지 보존 확인 (수치 적분 오차 관찰: 오일러 vs RK4)
- **M(q) 관성 행렬 변화**
  - q를 sweep하면서 M(q)의 대각 성분 변화 플롯
  - "이 자세에서 joint 2를 움직이려면 토크가 얼마나 필요한가" 직관

### ☆ 가치 보통
- **중력 보상 vs 미보상 비교**
  - τ = g(q) (중력 보상) vs τ = 0 (미보상) 시뮬레이션 나란히
  - 중력 보상 시 로봇이 자세를 유지하는 것 확인
- **코리올리/원심력 효과**
  - 한 관절을 빠르게 회전시킬 때 다른 관절에 미치는 커플링 토크
  - C(q,dq)·dq 벡터의 각 성분을 시계열로 표시
- **마찰 모델 효과** (추가)
  - 마찰 있음 vs 없음 시뮬레이션 비교
  - 쿨롱 마찰에 의한 stick-slip 현상 관찰
  - 기어비에 따른 반사 관성 증가 → 응답 변화
