# Chapter 3: Rigid-Body Motions

SO(3), SE(3) 및 관련 연산 구현. 다른 모든 챕터에서 import하는 기반 모듈.

## 구현 함수

| # | 함수 | 설명 |
|---|------|------|
| 3.1 | `RotInv(R)` | 회전행렬 역행렬 (= Rᵀ) |
| 3.2 | `Vec2so3(omega)` | 3벡터 → so(3) 반대칭 행렬 |
| 3.3 | `so32Vec(so3mat)` | so(3) → 3벡터 |
| 3.4 | `AxisAng3(expc3)` | 지수 좌표 → 회전축 + 회전각 |
| 3.5 | `MatrixExp3(hat_omega, theta)` | so(3) → SO(3) (로드리게스) |
| 3.6 | `MatrixLog3(R)` | SO(3) → so(3) |
| 3.7 | `Rp2Trans(R, p)` | R, p → T (4×4) |
| 3.8 | `Trans2Rp(T)` | T → R, p |
| 3.9 | `TransInv(T)` | T의 역행렬 |
| 3.10 | `Vec2se3(V)` | 6D 트위스트 → se(3) |
| 3.11 | `se32Vec(se3mat)` | se(3) → 6D 트위스트 |
| 3.12 | `Adjoint(T)` | 6×6 수반 표현 [Ad_T] |
| 3.13 | `Screw2Axis(q, s, h)` | 스크류 파라미터 → 스크류 축 S |
| 3.14 | `AxisAng(expc6)` | 6D 지수 좌표 → S, θ |
| 3.15 | `MatrixExp6(se3mat)` | se(3) → SE(3) |
| 3.16 | `MatrixLog(T)` | SE(3) → se(3) |

## 파일

| 파일 | 설명 |
|------|------|
| `modern_robotics_ch03.py` | 모듈 본체 (ch04~ch08에서 import) |
| `modern_robotics_ch03.ipynb` | 노트북 |
| `compared_mr2pin.py` | Pinocchio 비교 (exp3/log3, exp6/log6 등) |

## 검증

exp3, log3, exp6, log6, TransInv, Rp2Trans, AxisAng3, AxisAng 모두 Pinocchio와 PASS

## 실행

```bash
conda activate mr
python ch03_rigid_body_motion/modern_robotics_ch03.py
python ch03_rigid_body_motion/compared_mr2pin.py
```
