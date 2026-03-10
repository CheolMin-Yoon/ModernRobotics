# Chapter 11: 로봇 제어 (Robot Control)

## 구현 목록

### `pd_control.py` — PD 제어
- `pd_joint(q, dq, q_des, dq_des, Kp, Kd)`: 관절 공간 PD
  - τ = Kp(q_d - q) + Kd(dq_d - dq)
- 중력 보상 유무에 따른 성능 비교

### `ctc.py` — Computed Torque Control
- `ctc(robot, q, dq, q_des, dq_des, ddq_des, Kp, Kd)`:
  - τ = M(q)(ddq_d + Kp·e + Kd·ė) + C(q,dq)·dq + g(q)
- ch08의 동역학 함수 활용

### `impedance_control.py` — 임피던스 제어
- `impedance_joint(q, dq, q_eq, Md, Kd, Dd)`: 관절 임피던스
- `impedance_task(J, q, dq, x_eq, Md, Kd, Dd)`: 작업 공간 임피던스
- 외력 인가 시 컴플라이언스 시각화

### `task_space_control.py` — 작업 공간 제어 (추가)
- `task_space_mass(M, J)`: Λ(q) = (J M^{-1} J^T)^{-1}
- `dyn_consistent_pinv(M, J)`: J̄ = M^{-1} J^T Λ
- `task_space_ctc(robot, q, dq, x_des, dx_des, ddx_des, Kp, Kd)`: 작업 공간 CTC
- null-space 자세 최적화 동시 수행

### `adaptive_control.py` — 적응 제어 (추가)
- `adaptive_ctc(robot, q, dq, q_des, dq_des, ddq_des, Kp, Kd, param_est)`:
  - Y(q,dq,ddq)·π̂ 형태의 리그레서 기반 적응 제어
  - 파라미터 추정 업데이트: π̂_dot = -Γ · Y^T · s
- 질량 미지인 상태에서 적응적으로 보상하는 시뮬레이션
- 파라미터 수렴 과정 시각화

### `hybrid_control.py` — 하이브리드 힘/운동 제어
- 선택 행렬 S로 힘/위치 분리
- 벽면 접촉 시나리오 시뮬레이션

### `force_control.py` — 힘 제어 (추가)
- `explicit_force_control(F_des, F_meas, Kf)`: 명시적 힘 제어
- `admittance_control(F_ext, Md, Dd, Kd)`: 어드미턴스 제어
- 임피던스 vs 어드미턴스 제어 비교

### `sim_loop.py` — 제어 시뮬레이션 루프
- ch08 동역학 + ch09 궤적 + 제어기 통합
- MuJoCo 없이 순수 Python 시뮬 (오일러 적분)
- 응답 비교: PD vs CTC vs Impedance

### `mujoco_sim.py` — MuJoCo 연동 시뮬레이션 (추가)
- 3DOF 로봇 MJCF 로드 → 직접 구현한 제어기로 제어
- Python mujoco 패키지 사용
- 순수 Python 시뮬 결과와 MuJoCo 결과 비교

### Pinocchio 검증
- `pin.rnea()` 기반 CTC와 직접 구현 CTC 토크 비교

## 시각화

### ★ 가치 높음
- **제어기 응답 비교 (PD vs CTC vs Impedance)**
  - 같은 목표 궤적에 대해 세 제어기의 추종 성능 비교
  - 4단 플롯: q(t) 추종, e(t) 오차, τ(t) 토크, 에너지(t)
  - CTC의 모델 기반 보상 효과가 명확히 보임
- **외란 응답 시각화**
  - 특정 시각에 외력 인가 → 각 제어기의 복원 거동 비교
  - 임피던스 제어의 컴플라이언스 vs PD의 딱딱한 복원

### ☆ 가치 보통
- **게인 튜닝 효과**
  - Kp, Kd를 바꿔가며 응답 변화 (오버슈트, 정착 시간, 진동)
  - 2D 파라미터 맵: (Kp, Kd) → 정착 시간 히트맵
- **모델 오차 민감도**
  - CTC에서 질량을 ±20% 틀리게 줬을 때 추종 오차 변화
- **중력 보상 유무 비교**
  - PD only vs PD + gravity comp: 정상 상태 오차 차이
- **적응 제어 파라미터 수렴** (추가)
  - 추정 파라미터 π̂(t)가 실제 값으로 수렴하는 과정
  - 적응 제어 vs 고정 모델 CTC 추종 오차 비교
