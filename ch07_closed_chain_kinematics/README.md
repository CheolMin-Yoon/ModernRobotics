# Chapter 7: Kinematics of Closed Chains

폐연쇄 기구학 — 3지 그리퍼 파지 해석

## 구현 함수

| 절 | 함수 | 설명 |
|----|------|------|
| 7.1 | `grasp_matrix(p_contacts, p_obj)` | 파지 행렬 G (6×3k): 접촉력 → 물체 wrench |
| 7.2 | `hand_jacobian(finger_jacobians)` | 핸드 자코비안 J_h (블록 대각) |
| 7.3 | `partition_jacobian(J, active, passive)` | 능동/수동 관절 분리, H 행렬 |
| 7.3 | `closed_chain_jacobian(J_a, J_p, H)` | 폐연쇄 자코비안 J_closed = J_a + J_p·H |
| 7.4 | `check_force_closure(G)` | Force closure 판별 (SVD, isotropy index) |
| 7.5 | `grubler_dof(N, J, f_i, C)` | Grübler 공식: m = 6(N-1-J) + Σf_i - C |
| 7.6 | `three_finger_grasp_analysis(...)` | 종합 파지 해석 |

## 파일

| 파일 | 설명 |
|------|------|
| `modern_robotics_ch07.py` | 폐연쇄 기구학 함수 구현 |
| `modern_robotics_ch07_torch.py` | PyTorch 버전 (autograd 접촉점 최적화 포함) |
| `compared_mr2pin.py` | Pinocchio SE(3)/skew 기반 비교 검증 |
| `compared_mr2mujoco.py` | 수학적 검증 (해석적 기대값 비교) |

## PyTorch 버전

`modern_robotics_ch07_torch.py` — 파지 해석 + autograd 접촉점 최적화.

- 기존 함수 모두 torch 버전으로 구현 (numpy 교차 검증 완료)
- `optimize_contact_positions`: autograd로 파지 품질(σ_min/σ_max)을 최대화하는 접촉점 위치 최적화

## 검증

Pinocchio 비교 (`compared_mr2pin.py`):
- 파지 행렬 G (pin.skew 기반): PASS
- Force closure SVD: PASS
- Grübler DOF (6R, SCARA, 3-finger): PASS
- 핸드 자코비안 (block_diag): PASS
- 폐연쇄 자코비안 항등식: PASS
- skew 대칭성 + pin.skew 일치: PASS

수학적 검증 (`compared_mr2mujoco.py`):
- 파지 행렬 G 해석적 검증: PASS
- Force closure 알려진 케이스 (1점/3점/4점): PASS
- Grübler DOF 교재 예제: PASS
- 핸드 자코비안 블록 대각 구조: PASS
- H 행렬 + 폐연쇄 자코비안: PASS

## 실행

```bash
conda activate mr
python ch07_closed_chain_kinematics/modern_robotics_ch07.py
python ch07_closed_chain_kinematics/modern_robotics_ch07_torch.py
python ch07_closed_chain_kinematics/compared_mr2pin.py
python ch07_closed_chain_kinematics/compared_mr2mujoco.py
```
