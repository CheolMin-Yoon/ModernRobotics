# Chapter 3: 강체 운동 (Rigid Body Motion)

## 구현 목록

### `rotation.py` — 회전 표현
- `rot_x(theta)`, `rot_y(theta)`, `rot_z(theta)`: 기본 회전 행렬
- `so3_to_vec(so3mat)`: 3x3 skew-symmetric → 3-vector (ω)
- `vec_to_so3(omega)`: 3-vector → 3x3 skew-symmetric [ω]
- `axis_angle_to_rot(omega_hat, theta)`: 축-각 → R (Rodrigues' formula)
- `rot_to_axis_angle(R)`: R → (ω_hat, θ)
- `matrix_exp3(so3mat)`: so(3) → SO(3) 지수 사상
- `matrix_log3(R)`: SO(3) → so(3) 로그 사상

### `euler_angles.py` — 오일러각 표현 (추가)
- `euler_zyz_to_rot(alpha, beta, gamma)`: ZYZ 오일러각 → R
- `rot_to_euler_zyz(R)`: R → ZYZ 오일러각
- `euler_zyx_to_rot(yaw, pitch, roll)`: ZYX (RPY) → R
- `rot_to_euler_zyx(R)`: R → ZYX
- gimbal lock 발생 조건 확인 및 시연

### `quaternion.py` — 쿼터니언 표현 (추가)
- `quat_from_rot(R)`: 회전 행렬 → unit quaternion [w, x, y, z]
- `rot_from_quat(q)`: unit quaternion → 회전 행렬
- `quat_multiply(q1, q2)`: 쿼터니언 곱
- `quat_slerp(q1, q2, t)`: 구면 선형 보간 (SLERP)
- `quat_from_axis_angle(axis, angle)`: 축-각 → 쿼터니언
- Pinocchio 내부 쿼터니언과 비교 검증

### `transform.py` — 동차 변환
- `rp_to_trans(R, p)`: (R, p) → 4x4 T
- `trans_to_rp(T)`: T → (R, p)
- `trans_inv(T)`: T의 역변환
- `vec_to_se3(V)`: 6-vector twist → 4x4 [V]
- `se3_to_vec(se3mat)`: 4x4 → 6-vector
- `adjoint(T)`: 6x6 Adjoint 행렬 [Ad_T]
- `matrix_exp6(se3mat)`: se(3) → SE(3) 지수 사상
- `matrix_log6(T)`: SE(3) → se(3) 로그 사상

### Pinocchio 검증
- `pinocchio.SE3`와 직접 구현 결과 비교
- `pin.log3()`, `pin.exp3()` 결과와 대조
- `pin.Quaternion` ↔ 직접 구현 쿼터니언 비교

## 시각화

### ★ 가치 높음
- **회전 보간 애니메이션**
  - R_start → R_end를 matrix_exp3(log3(R) · t)로 보간
  - 3D 좌표 프레임이 부드럽게 회전하는 애니메이션
  - 오일러각 보간 vs 지수 사상 보간 vs SLERP 비교 (gimbal lock 시연)
- **스크류 운동 시각화**
  - 6D twist → 스크류 축 (3D 직선) + 회전 방향 + 병진 방향 화살표
  - 물체가 스크류 축을 따라 나선형으로 이동하는 애니메이션
- **gimbal lock 시연** (추가)
  - ZYZ 오일러각으로 보간할 때 β ≈ 0 근처에서 경로가 꼬이는 것
  - 같은 보간을 쿼터니언 SLERP로 하면 매끄러운 것 비교

### ☆ 가치 보통
- **SO(3) 지수/로그 사상 대응**
  - so(3)의 [ω]θ (skew-symmetric) → SO(3)의 R 변환 과정을 단계별 시각화
  - 축-각 표현: 회전축 벡터 + 각도 원호를 3D로 표시
- **Adjoint 변환 효과**
  - 같은 twist를 다른 프레임에서 봤을 때 어떻게 바뀌는지
  - 두 좌표 프레임 + twist 화살표를 동시에 표시
