# Chapter 8: Dynamics of Open Chains

개방 연쇄 동역학 — UR5e 기반 (SI 단위, m/kg)

## 대상 로봇

- UR5e (ros-industrial URDF 기준)

## UR5e 파라미터 (URDF → 수동 정의)

- 링크 길이: L1_e, L2_e, W1_e, W2_e, H1_e, H2_e
- 스크류 축: URDF joint rpy를 `Rot_rpy()` 누적 회전으로 world frame 축 계산
- 동역학: 질량 m (6 링크), CoM r, 관성 텐서 I_b (rpy 반영)
- upper_arm, forearm은 URDF inertial rpy=(0,π/2,0) → Ixx↔Izz 교환

## 구현 함수

| 절 | 함수 | 설명 |
|----|------|------|
| 8.2 | `spatial_inertia(I_b, m)` | 6×6 공간 관성 행렬 G_b |
| 8.2 | `lie_bracket(V_b)` | 6×6 리 브라켓 [ad_V] |
| 8.2 | `calculate_wrench(G_b, dV_b, V_b)` | 렌치 F_b = G_b·dV - [ad_V]^T·G_b·V |
| 8.2 | `transform_to_space(T_sb, G_b, V_b, F_b)` | 물체→공간 좌표계 변환 |
| 8.3 | `RNEA(...)` | 역 뉴턴-오일러 (스켈레톤, 구현 예정) |

## 파일

- `modern_robotics_ch08.py` — UR5e 파라미터 + 동역학 함수
- `modern_robotics_ch08.ipynb` — 노트북
- `compared_mr2pin.py` — Pinocchio (UR5e URDF) 비교: 공간 관성 행렬 G_b
- `compared_mr2mujoco.py` — MuJoCo (UR5e scene) 비교: 관성, 질량 행렬, 역동역학, 중력 토크

## 검증

- 공간 관성 행렬 G_b: 6개 링크 모두 Pinocchio와 PASS
- MuJoCo 비교: 관성 파라미터, M(q), mj_inverse 토크, 중력 토크, nle 출력
