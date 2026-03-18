# Chapter 4: Forward Kinematics

Modern Robotics 4장 구현 - PoE(Product of Exponentials) 기반 순운동학

## 대상 로봇

- UR5 6DOF (책 222p 파라미터 기준)
- UR5e 파라미터는 주석 처리로 전환 가능

## 구현 내용

| 번호 | 함수/항목 | 설명 |
|------|-----------|------|
| - | 로봇 파라미터 | L1, L2, W1, W2, H1, H2 및 영 위치 M 행렬 |
| - | 공간꼴 스크류 | w, q, v = -w×q 로 S_i 구성 |
| - | 물체꼴 스크류 | B_i = [Ad_{M⁻¹}] · S_i (ch03 Adjoint 활용) |
| 4.1 | `body_frame_fk(Blist, thetalist)` | 물체꼴 PoE FK: T = M · e^{[B1]θ1} · ... · e^{[Bn]θn} |
| 4.2 | `fixed_frame_fk(Slist, thetalist)` | 공간꼴 PoE FK: T = e^{[S1]θ1} · ... · e^{[Sn]θn} · M |

## 파일

- `modern_robotics_ch04.py` - PoE FK 구현 (ch03 함수 import)
- `compared_mr2pin.py` - Pinocchio (UR5 URDF) 대비 FK 검증
- `modern_robotics_ch04.ipynb` - 노트북

## 의존성

- ch03 함수: `Adjoint`, `Vec2so3`, `Vec2se3`, `TransInv`
- `scipy.linalg.expm` (행렬 지수)
- 검증용: `pinocchio`, UR5 URDF (`urdf_files_dataset/`)

## 검증 결과 (compared_mr2pin.py)

```
conda activate mr
python compared_mr2pin.py
```

zero config, theta2=-90/theta5=90, all 45deg, random config 4가지 설정에서
space FK, body FK 모두 Pinocchio FK와 PASS (diff < 1e-10)
