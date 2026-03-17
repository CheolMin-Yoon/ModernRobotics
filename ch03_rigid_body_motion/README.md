# Chapter 3: Rigid-Body Motions

Modern Robotics 3장 구현 - SO(3), SE(3) 및 관련 연산

## 구현 함수 목록

| 번호 | 함수명 | 설명 |
|------|--------|------|
| 3.1 | `RotInv(R)` | 회전행렬 R의 역행렬 (= R^T) |
| 3.2 | `Vec2so3(omega)` | 3벡터 → so(3) 반대칭 행렬 |
| 3.3 | `so32Vec(so3mat)` | so(3) 반대칭 행렬 → 3벡터 |
| 3.4 | `AxisAng3(expc3)` | 지수 좌표 → 회전축 + 회전각 |
| 3.5 | `MatrixExp3(hat_omega, theta)` | so(3) 행렬지수 → SO(3) 회전행렬 (로드리게스 공식) |
| 3.6 | `MatrixLog3(R)` | SO(3) 행렬로그 → so(3) |
| 3.7 | `Rp2Trans(R, p)` | R, p → 동차 변환행렬 T |
| 3.8 | `Trans2Rp(T)` | 동차 변환행렬 T → R, p |
| 3.9 | `TransInv(T)` | 동차 변환행렬 T의 역행렬 |
| 3.10 | `Vec2se3(V)` | 6D 트위스트 → se(3) 행렬 |
| 3.11 | `se32Vec(se3mat)` | se(3) 행렬 → 6D 트위스트 |
| 3.12 | `Adjoint(T)` | 동차 변환행렬 T의 6x6 수반 표현 [Ad_T] |
| 3.13 | `Screw2Axis(q, s, h)` | 스크류 파라미터 → 정규화된 스크류 축 S |
| 3.14 | `AxisAng(expc6)` | 6D 지수 좌표 → 스크류 축 S + 변위 theta |
| 3.15 | `MatrixExp6(se3mat)` | se(3) 행렬지수 → SE(3) 변환행렬 |
| 3.16 | `MatrixLog6(T)` | SE(3) 행렬로그 → se(3) |

## 파일

- `modern_robotics_ch03.ipynb` - 노트북 (셀별 구현)
- `modern_robotics_ch03.py` - 모듈 (다른 챕터에서 import용)

## 사용법

다른 챕터에서 함수를 가져다 쓸 때:

```python
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ch03_rigid_body_motion.modern_robotics_ch03 import *
```
