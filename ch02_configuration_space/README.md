# Chapter 2: Configuration Space

형상 공간(C-space), 위상 공간, 자유도, 제약조건

## 구현 내용

| 파일 | 설명 |
|------|------|
| `topology.py` | 기본 위상 공간 (Eⁿ, S¹, Tⁿ), T² 토러스 시각화 |
| `Explicit_Representation.py` | S¹의 명시적 표현 (θ → cos, sin) |
| `Implicit_Representation.py` | S¹의 암묵적 표현 (x²+y²-1=0 등고선) |
| `constraints.py` | 홀로노믹/파피안(논홀로노믹) 제약조건 예제 |
| `c_space.py` | Grübler 공식, 2-링크 C-space 시각화 |

## 주요 함수

- `gruebler_formula(m, N, J, f_i)` — 자유도 계산: dof = m(N-1-J) + Σf_i
- `g_holonomic(theta)` — 홀로노믹 제약 예제 (원 위의 점)
- `check_pfaffian_constraint(theta, theta_dot)` — 파피안 제약 예제 (유니사이클)

## 실행

```bash
conda activate mr
python ch02_configuration_space/c_space.py
python ch02_configuration_space/topology.py
```
