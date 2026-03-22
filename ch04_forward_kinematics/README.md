# Chapter 4: Forward Kinematics

PoE(Product of Exponentials) 기반 순운동학

## 대상 로봇

- UR5 6DOF (교재 222p, 단위: mm)
- UR5e 6DOF (`modern_robotics_ch04_ur5e.py`, 단위: m, `params/ur5e.py` 참조)

## 구현 내용

| 함수 | 설명 |
|------|------|
| `body_frame_fk(Blist, θ, M)` | 물체꼴 PoE FK: T = M · exp([B₁]θ₁) · ... · exp([Bₙ]θₙ) |
| `fixed_frame_fk(Slist, θ, M)` | 공간꼴 PoE FK: T = exp([S₁]θ₁) · ... · exp([Sₙ]θₙ) · M |

## 파일

| 파일 | 설명 |
|------|------|
| `modern_robotics_ch04.py` | UR5 FK 구현 |
| `modern_robotics_ch04_ur5e.py` | UR5e FK (params/ur5e.py 파라미터 사용) |
| `modern_robotics_ch04.ipynb` | 노트북 |
| `compared_mr2pin.py` | Pinocchio (UR5 URDF) 비교 |
| `compared_mr2mujoco.py` | MuJoCo (UR5e scene) 비교 |

## 검증

4가지 config (zero, θ₂=-90°/θ₅=90°, all 45°, random)에서 space/body FK 모두 Pinocchio, MuJoCo와 PASS

## 실행

```bash
conda activate mr
python ch04_forward_kinematics/modern_robotics_ch04.py
python ch04_forward_kinematics/compared_mr2pin.py
python ch04_forward_kinematics/compared_mr2mujoco.py
```
