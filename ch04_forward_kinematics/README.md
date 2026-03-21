# Chapter 4: Forward Kinematics

PoE(Product of Exponentials) 기반 순운동학

## 대상 로봇

- UR5 6DOF (책 222p, 단위: mm)

## 구현 내용

| 항목 | 설명 |
|------|------|
| 로봇 파라미터 | L1, L2, W1, W2, H1, H2, 영 위치 M |
| 공간꼴 스크류 | w, q, v = -w×q → S_i |
| 물체꼴 스크류 | B_i = [Ad_{M⁻¹}] · S_i |
| `body_frame_fk(Blist, θ)` | 물체꼴 PoE FK |
| `fixed_frame_fk(Slist, θ)` | 공간꼴 PoE FK |

## 파일

- `modern_robotics_ch04.py` — FK 구현
- `modern_robotics_ch04.ipynb` — 노트북
- `compared_mr2pin.py` — Pinocchio (UR5 URDF) 비교
- `compared_mr2mujoco.py` — MuJoCo (UR5e scene) 비교

## 검증

4가지 config (zero, θ2=-90/θ5=90, all 45°, random)에서 space/body FK 모두 Pinocchio, MuJoCo와 PASS
