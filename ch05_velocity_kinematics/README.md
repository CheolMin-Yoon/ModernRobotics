# Chapter 5: Velocity Kinematics and Statics

물체/공간 자코비안 구현

## 구현 함수

| # | 함수 | 설명 |
|---|------|------|
| 5.1 | `BodyJacobian(Blist, θ)` | 물체 자코비안 J_b (6×n) |
| 5.2 | `SpaceJacobian(Slist, θ)` | 공간 자코비안 J_s (6×n) |

## 파일

- `modern_robotics_ch05.py` — 자코비안 구현
- `compared_mr2pin.py` — Pinocchio (UR5 URDF) 비교
- `compared_mr2mujoco.py` — MuJoCo (UR5e scene) 비교

## 검증

5가지 config에서 J_s, J_b 모두 Pinocchio, MuJoCo와 PASS.
`J_s = [Ad_Tsb] @ J_b` 관계도 검증 완료.
