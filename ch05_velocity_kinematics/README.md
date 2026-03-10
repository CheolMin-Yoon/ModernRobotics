# Chapter 5: 속도 기구학과 정역학 (Velocity Kinematics and Statics)

## 구현 목록

### `jacobian.py` — 자코비안
- `body_jacobian(Blist, thetalist)`: Body Jacobian J_b
- `space_jacobian(Slist, thetalist)`: Space Jacobian J_s
- 매니퓰러빌리티 타원체 시각화
  - `manipulability_ellipsoid(J)`: σ_i 계산 + 3D 타원체 plot

### `manipulability.py` — 매니퓰러빌리티 지표 (추가)
- `manipulability_index(J)`: √det(J·J^T) — 볼륨 지표
- `condition_number(J)`: σ_max / σ_min — 등방성 지표
- `min_singular_value(J)`: σ_min — 특이점까지 거리
- 세 지표를 workspace 위에 색상맵으로 비교

### `nullspace.py` — Null-space 프로젝션 (추가)
- `null_space_projector(J)`: N = I - J†·J
- `redundancy_resolution(J, dq_0)`: null-space 속도 활용
- 7DOF 로봇에서 EE 고정 + 팔꿈치 이동 시연
- 관절 제한 회피를 null-space 목적함수로 구현

### `statics.py` — 정역학
- `wrench_to_joint_torque(J, F)`: τ = J^T · F
- 주어진 end-effector wrench에 대한 관절 토크 계산

### `singularity.py` — 특이점 분석
- `det_jacobian(J)`: det(J) 또는 det(J·J^T) 계산
- 관절 공간 sweep → 특이점 맵 시각화

### Pinocchio 검증
- `pin.computeJointJacobians()` 결과와 비교
- `pin.getFrameJacobian()` (LOCAL, WORLD, LOCAL_WORLD_ALIGNED) 프레임 차이 확인

## 시각화

### ★ 가치 높음
- **매니퓰러빌리티 타원체**
  - EE 위치에 속도 타원체 (J·J^T의 고유값 → 반축 길이) 3D 표시
  - 힘 타원체 (J^{-T}·J^{-1}) 동시 표시 → 속도 잘 나는 방향 = 힘 약한 방향 확인
  - q를 바꿔가며 타원체 형태 변화 관찰
- **workspace + manipulability 색상맵**
  - 관절 공간 대량 샘플링 → FK → EE 점군
  - 색상 = manipulability index (√det(J·J^T))
  - 특이점 근처가 어두워지는 것 확인

### ☆ 가치 보통
- **특이점 맵 히트맵**
  - 2-link planar: (θ1, θ2) 평면에서 det(J) 값을 히트맵으로 표시
  - 3DOF: θ1 고정, (θ2, θ3) 슬라이스별 det(J) 히트맵
  - 특이점 곡선/면이 어디에 있는지 시각적으로 확인
- **정역학 wrench 시각화**
  - EE에 가해지는 외력 F → τ = J^T·F 관절 토크 바 차트
  - 같은 힘이라도 자세에 따라 필요 토크가 달라지는 것 비교
