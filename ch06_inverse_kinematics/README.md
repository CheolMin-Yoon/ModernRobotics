# Chapter 6: 역기구학 (Inverse Kinematics)

## 구현 목록

### `ik_newton.py` — Newton-Raphson IK
- `ik_body(Blist, M, T_target, q0, tol, max_iter)`: Body frame IK
  - e = log(T_target^{-1} · T_current) → Δθ = J_b^{-1} · e 반복
- `ik_space(Slist, M, T_target, q0, tol, max_iter)`: Space frame IK

### `ik_dls.py` — Damped Least Squares IK
- `ik_dls(J, error, lambda_)`: Δθ = J^T(JJ^T + λ²I)^{-1} e
- 특이점 근처 안정성 비교 (Newton vs DLS)

### `ik_analytical.py` — 해석적 IK (추가)
- `ik_3dof_geometric(robot, target_pos)`: 3DOF RRR 기하학적 풀이
  - 코사인 법칙 기반 elbow angle 계산
  - elbow-up / elbow-down 두 해 반환
- 수치 IK 결과와 해석적 IK 결과 비교 검증

### `ik_constrained.py` — 관절 제한 IK (추가)
- `ik_with_limits(J, error, q, q_min, q_max, lambda_)`: 관절 제한 반영
  - clamping 방식: Δq 적용 후 q_min/q_max로 클램핑
  - null-space projection 방식: 관절 제한 회피를 null-space 목적함수로
  - weighted DLS: 관절 제한 근처에서 가중치 증가

### `ik_analysis.py` — IK 분석 도구
- 수렴 이력 시각화 (iteration vs error)
- workspace 시각화: 도달 가능 영역 plot
- 다중 해 탐색: 여러 초기값에서 IK → 해 비교
- IK 실패 케이스 분석 (추가):
  - workspace 밖 목표 → 발산 과정
  - 특이점 위의 목표 → Newton 발산 vs DLS 수렴
  - 관절 제한에 걸리는 경우 → constrained IK 효과

### Pinocchio 검증
- 직접 구현 IK 결과 q → `pin.forwardKinematics(q)` → T 비교

## 시각화

### ★ 가치 높음
- **IK 수렴 과정 애니메이션**
  - 매 iteration마다 로봇 자세를 그려서 EE가 목표로 수렴하는 과정 시각화
  - iteration vs position error 그래프 동시 표시
  - Newton vs DLS vs Analytical 수렴 속도 비교
- **workspace 도달 범위**
  - 관절 공간 대량 샘플링 → FK → EE 점군 (3D scatter)
  - IK 목표점이 workspace 안/밖인지 시각적으로 판단
  - 도달 불가 영역에 IK 시도 → 발산 과정 시각화

### ☆ 가치 보통
- **다중 해 시각화**
  - 같은 EE 목표에 대해 여러 초기값 → 서로 다른 IK 해
  - 각 해에 해당하는 로봇 자세를 반투명하게 겹쳐 표시
  - elbow-up / elbow-down 등 해의 분류
- **damping λ 효과**
  - DLS에서 λ 값에 따른 수렴 속도 / 정확도 트레이드오프
  - λ = 0 (순수 Newton) vs λ = 0.01 vs λ = 0.1 비교 그래프
