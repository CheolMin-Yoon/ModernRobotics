# Chapter 6: Inverse Kinematics

수치 역기구학 (Newton-Raphson 기반)

## 구현 함수

| 함수 | 설명 |
|------|------|
| `IKinBody(Blist, M, T_sd, θ₀, ew, ev)` | 물체꼴 수치 IK |
| `IKinSpace(Slist, M, T_sd, θ₀, ew, ev)` | 공간꼴 수치 IK |

알고리즘: 매 반복마다 `V = se32Vec(MatrixLog(T_bs⁻¹ T_sd))` 오차를 계산하고 `J†V`로 관절각 갱신.

## 파일

| 파일 | 설명 |
|------|------|
| `modern_robotics_ch06.py` | IK 구현 |
| `compared_mr2pin.py` | Pinocchio IK 비교 |
| `compared_mr2mujoco.py` | MuJoCo IK 비교 |

## 참고

관절 제한 + 질량 행렬 가중치가 필요한 경우 `kinematics_pick_and_place/osqp_ik.py` 참조 (OSQP QP 기반 IK).

## 실행

```bash
conda activate mr
python ch06_inverse_kinematics/modern_robotics_ch06.py
python ch06_inverse_kinematics/compared_mr2pin.py
```
