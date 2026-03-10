# Chapter 7: 폐연쇄의 기구학 (Closed-Chain Kinematics)

## 구현 목록

### `parallel_mechanism.py` — 병렬 메커니즘
- 3xRPR 평면 병렬 로봇 FK/IK
- Stewart-Gough 플랫폼 기구학 (6DOF)

### `four_bar.py` — 4절 링크 메커니즘 (추가)
- `four_bar_fk(theta_input, link_lengths)`: 입력각 → 출력각
- `four_bar_jacobian(theta, link_lengths)`: 속도 관계
- Grashof 조건 판별

### `constraint.py` — 구속 조건
- `loop_closure(T_list)`: 루프 폐합 조건 T1·T2·...·Tn = I
- `constraint_jacobian(robot, q)`: 구속 자코비안
- 자코비안 기반 구속 해석

### `hybrid_mechanism.py` — 직병렬 혼합 메커니즘 (추가)
- 직렬 + 병렬 조합 기구학
- 구속 조건 하에서의 자유도 계산

이 챕터는 개연쇄 이해 후 확장용. 우선순위 낮음

## 시각화

### ☆ 가치 보통
- **4절 링크 애니메이션**
  - 입력 크랭크 회전 → 나머지 링크 운동 애니메이션
  - Grashof / non-Grashof 조건에 따른 운동 범위 차이
- **Stewart 플랫폼**
  - 6개 다리 길이 변화 → 플랫폼 자세 변화 3D 애니메이션
