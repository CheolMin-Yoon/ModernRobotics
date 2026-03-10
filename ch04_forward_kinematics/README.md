# Chapter 4: 정기구학 (Forward Kinematics)

## 구현 목록

### `poe_fk.py` — Product of Exponentials 정기구학
- `fk_in_body(M, Blist, thetalist)`: Body frame PoE FK
  - M: 홈 포지션 end-effector 변환
  - Blist: body frame 스크류 축 리스트
  - T = M · e^{[B1]θ1} · e^{[B2]θ2} · ...
- `fk_in_space(M, Slist, thetalist)`: Space frame PoE FK
  - T = e^{[S1]θ1} · e^{[S2]θ2} · ... · M

### `dh_fk.py` — DH Convention FK (비교용)
- 기존 `robot.py`의 DH FK를 여기로 정리
- PoE FK 결과와 DH FK 결과 비교

### `robot_configs.py` — 로봇 설정
- 3DOF RRR: M, Slist, Blist 정의
- UR5e 6DOF: M, Slist, Blist 정의 (Pinocchio URDF에서 추출)

### `urdf_to_poe.py` — URDF → PoE 자동 추출 (추가)
- `extract_screw_axes(urdf_path, ee_frame)`: URDF 파싱 → Slist, M 자동 생성
- Pinocchio model에서 관절 축/위치 추출 → 스크류 축 계산
- 수동 정의한 값과 자동 추출 값 비교 검증

### Pinocchio 검증
- `pin.forwardKinematics(model, data, q)` 결과와 PoE FK 비교
- 다양한 q에서 오차 확인

## 시각화

### ★ 가치 높음
- **FK 인터랙티브 슬라이더**
  - matplotlib slider로 q1, q2, q3 조절 → 실시간 로봇 자세 업데이트
  - 각 관절 프레임의 좌표축 (RGB = XYZ) 표시
  - EE 위치/자세 텍스트 오버레이
- **PoE vs DH 비교**
  - 같은 q에서 PoE FK와 DH FK 결과를 나란히 표시
  - 두 방법의 중간 프레임 차이 시각화

### ☆ 가치 보통
- **스크류 축 시각화**
  - 각 관절의 space frame 스크류 축 S_i를 3D 공간에 직선 + 화살표로 표시
  - q가 바뀌면 body frame 스크류 축 B_i가 어떻게 이동하는지 애니메이션
