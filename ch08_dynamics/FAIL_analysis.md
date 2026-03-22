# ch08 동역학 비교 검증 FAIL 분석

## 1. 코드 버그: `VelQuadraticForces` — `NameError: n is not defined`

**파일**: `ch08_dynamics/modern_robotics_ch08.py`, line 213

```python
def VelQuadraticForces(thetalist, dthetalist, Mlist, Glist, Slist):
    ddthetalist_zero = np.zeros(n)  # ← n 미정의
```

`n = len(thetalist)` 한 줄이 빠져 있음. `GravityForces`에는 있는데 `VelQuadraticForces`에만 누락됨.
→ 이 함수를 호출하는 [5] 비선형 효과, [6] 분해 검증이 모두 실행 자체가 안 됨.

---

## 2. 중력 토크 g(q) FAIL — 부호 반전 + 값 불일치

**비교**: MR vs Pinocchio / MR vs MuJoCo 모두 FAIL

| config | MR 결과 | Pinocchio/MuJoCo | diff |
|---|---|---|---|
| home config | `[0, 0, 0, 0, 0, 0]` | `[0, -1.38, -1.38, -1.38, 0, 0]` | 2.38 |
| zero config | `[0, +52.4, +14.5, 0, 0, 0]` | `[0, -52.4, -14.5, 0, 0, 0]` | 109 |
| random config | `[0, +27.0, +13.2, -0.13, 0.08, 0]` | `[0, -28.2, -14.5, -1.16, 0.05, 0]` | 61.8 |

**원인 분석**:

`params/ur5e.py`의 `Glist` 구성 방식 문제.

현재 `Glist`는 MR 링크 프레임(z축 = w_i, 원점 = q_i) 기준으로 URDF 관성을 변환 후 CoM 오프셋 평행축 정리를 적용해서 만들고 있음.

그런데 MR 교재 Algorithm 8.2의 RNEA는 `G_i`가 **링크 i의 CoM 프레임** 기준이고, `Mlist[i]`도 **CoM 프레임 간 변환**이어야 함. 현재는 조인트 프레임 기준 `Mlist`와 CoM 오프셋이 포함된 `Glist`가 혼용되어 있어서 CoM 위치가 이중으로 반영되거나 누락됨.

특히 `zero config`에서 부호가 완전히 반전되는 것은 CoM 오프셋 방향이 MR 프레임 변환 과정에서 뒤집히는 것으로 보임.

---

## 3. 질량 행렬 M(q) FAIL

**비교**: MR vs Pinocchio / MR vs MuJoCo 모두 FAIL

| config | diff (vs Pinocchio) | diff (vs MuJoCo) |
|---|---|---|
| home config | 2.60e-01 | 4.02e-01 |
| zero config | 7.16e-02 | 2.82e-01 |
| random config | 2.90e-01 | 4.64e-01 |

**원인 분석**:

`MassMatrix`는 내부적으로 `RNEA`를 n번 호출하므로, `Glist`/`Mlist` 불일치가 그대로 전파됨.
MuJoCo 결과와 비교하면 대각 원소 차이가 크고, 특히 wrist 관절(4~6번)의 off-diagonal 항이 다름.
이것도 CoM 오프셋이 제대로 반영되지 않아서 링크 간 관성 결합 항이 틀리게 계산되는 것.

---

## 4. 관성 파라미터 G_b FAIL (MuJoCo 비교 시)

**비교**: MR vs MuJoCo

| 링크 | 상태 | 원인 |
|---|---|---|
| shoulder_link | PASS | CoM = 0, 단순 케이스 |
| upper_arm_link | FAIL (diff=1.68e-01) | MR I_diag = [0.015, 0.134, 0.134], MuJoCo = [0.134, 0.134, 0.015] → 축 순서 다름 |
| forearm_link | FAIL (diff=3.83e-02) | 동일하게 축 순서 불일치 |
| wrist_1~2_link | PASS | 대칭 관성이라 순서 무관 |
| wrist_3_link | FAIL (diff=1.73e-03) | 질량 차이 (MR=0.1879, MuJoCo=0.1889) + 축 순서 불일치 |

**원인**: `params/ur5e.py`에서 `I_b = inertia`로 정의된 관성 텐서의 축 순서(Ixx, Iyy, Izz)가 MuJoCo가 저장하는 축 순서와 다름. MuJoCo는 body 로컬 관성 주축 기준으로 저장하고 `body_iquat`으로 회전을 별도 관리함.

---

## 요약

| 항목 | 상태 | 원인 |
|---|---|---|
| `VelQuadraticForces` | 코드 버그 | `n = len(thetalist)` 누락 |
| 중력 토크 g(q) | FAIL | `Glist`/`Mlist` CoM 프레임 불일치 |
| 질량 행렬 M(q) | FAIL | 동일 원인 (RNEA 기반) |
| RNEA 토크 | FAIL | 동일 원인 |
| 관성 파라미터 G_b | 일부 FAIL | MuJoCo 축 순서 + 질량 값 차이 |
| 비선형 효과 h(q,dq) | 실행 불가 | 버그 #1로 인한 크래시 |
| 분해 검증 | 실행 불가 | 버그 #1로 인한 크래시 |

**핵심 수정 필요 사항**:
1. `VelQuadraticForces`에 `n = len(thetalist)` 추가 (단순 버그)
2. `params/ur5e.py`의 `Mlist`와 `Glist`를 MR 교재 Algorithm 8.2 convention에 맞게 재구성
   - `G_i`: CoM 프레임 기준 `[[I_com, 0], [0, mI]]` (평행축 정리 없이)
   - `Mlist[i]`: 링크 i-1 CoM 프레임 → 링크 i CoM 프레임 변환
