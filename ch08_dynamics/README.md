# Chapter 8: Dynamics of Open Chains

개방 연쇄 동역학 — UR5e 기반 (SI 단위, m/kg)

## 대상 로봇

- UR5e (ros-industrial URDF 기준, `params/ur5e.py`)

## 구현 함수

| 절 | 함수 | 설명 |
|----|------|------|
| 8.2 | `spatial_inertia(I_b, m)` | 6×6 공간 관성 행렬 G_b |
| 8.2 | `lie_bracket(V_b)` | 6×6 리 브라켓 [ad_V] |
| 8.2 | `calculate_wrench(G_b, dV_b, V_b)` | 렌치 F_b = G_b·dV - [ad_V]ᵀ·G_b·V |
| 8.2 | `transform_to_space(T_sb, G_b, V_b, F_b)` | 물체→공간 좌표계 변환 |
| 8.3 | `RNEA(thetalist, dthetalist, ddthetalist, g, F_tip, Mlist, Glist, Slist)` | 역 뉴턴-오일러 (Algorithm 8.2) |
| 8.3 | `MassMatrix(thetalist, Mlist, Glist, Slist)` | 질량 행렬 M(θ) — RNEA n회 호출 |
| 8.3 | `MassMatrixCRBA(thetalist, Mlist, Glist, Slist)` | 질량 행렬 M(θ) — CRBA O(n²) |
| 8.3 | `VelQuadraticForces(thetalist, dthetalist, Mlist, Glist, Slist)` | 코리올리 항 c(θ,dθ) |
| 8.3 | `GravityForces(thetalist, g, Mlist, Glist, Slist)` | 중력 토크 g(θ) |

## 파일

| 파일 | 설명 |
|------|------|
| `modern_robotics_ch08.py` | 동역학 함수 구현 |
| `modern_robotics_ch08.ipynb` | 노트북 |
| `compared_mr2pin.py` | Pinocchio (UR5e URDF) 비교 |
| `compared_mr2mujoco.py` | MuJoCo (UR5e scene) 비교 |
| `FAIL_analysis.md` | 현재 검증 FAIL 항목 분석 |

## 검증 현황

| 항목 | Pinocchio | MuJoCo |
|------|-----------|--------|
| 공간 관성 행렬 G_b | ✓ PASS (6/6) | 일부 FAIL (축 순서 불일치) |
| 질량 행렬 M(q) | ✗ FAIL | ✗ FAIL |
| 역동역학 RNEA | ✗ FAIL | ✗ FAIL |
| 중력 토크 g(q) | ✗ FAIL | ✗ FAIL |
| 비선형 효과 h(q,dq) | 크래시 | 크래시 |

FAIL 원인 및 상세 분석 → `FAIL_analysis.md` 참조

## 실행

```bash
conda activate mr
python ch08_dynamics/modern_robotics_ch08.py
python ch08_dynamics/compared_mr2pin.py
python ch08_dynamics/compared_mr2mujoco.py
```
