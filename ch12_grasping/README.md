# Chapter 12: 파지와 조작 (Grasping and Manipulation)

## 구현 목록

### `contact.py` — 접촉 모델
- `point_contact(normal, mu)`: 점 접촉 wrench cone
- `friction_cone(mu, n_edges)`: 마찰 원뿔 다면체 근사
- 접촉 유형: 점, 선, 면 접촉 wrench 공간

### `friction.py` — 마찰 모델 세부 (추가)
- `sliding_friction(v_t, mu, fn)`: 미끌림 마찰
- `rolling_friction(omega, mu_r, fn)`: 구름 마찰
- `torsional_friction(omega_n, mu_t, fn)`: 비틀림 마찰
- 접촉 모드 판별: stick / slide / separate

### `grasp_analysis.py` — 파지 분석
- `grasp_matrix(contacts)`: Grasp matrix G 계산
- `force_closure(G)`: 힘 폐합 판정
- `form_closure(contacts)`: 형태 폐합 판정
- form closure vs force closure 차이 예제 (추가)
  - form closure: 마찰 없이 구속
  - force closure: 마찰 포함 구속

### `grasp_quality.py` — 파지 품질 지표 (추가)
- `min_wrench(G, mu)`: 최소 wrench 크기 (worst-case 저항력)
- `grasp_isotropy(G)`: 파지 등방성 지표
- `largest_min_resisted_wrench(G, friction_cones)`: 최대 최소 저항 wrench
- 접촉점 배치 최적화: 품질 지표 최대화

### `grasp_viz.py` — 시각화
- 2D 물체 + 접촉점 + 마찰 원뿔 시각화
- wrench 공간 시각화

### Pinocchio 연계
- FrJoCo_C의 3-finger gripper 파라미터 활용 가능

## 시각화

### ★ 가치 높음
- **2D 파지 + 마찰 원뿔 + force closure**
  - 2D 물체 (원, 사각형) + 접촉점 위치 + 법선 방향
  - 각 접촉점에서 마찰 원뿔을 부채꼴로 표시
  - wrench 공간 (fx, fy, τ) 3D에서 원뿔들의 convex hull
  - 원점 포함 여부 → force closure 판정 결과 표시
- **접촉점 이동에 따른 force closure 변화**
  - 접촉점 위치를 인터랙티브하게 이동 → 실시간 force closure 판정

### ☆ 가치 보통
- **마찰 계수 μ 변화 효과**
  - μ를 0.1 → 1.0으로 바꿔가며 마찰 원뿔 크기 변화
  - force closure가 성립하는 최소 μ 탐색
- **3-finger vs 2-finger 비교**
  - 같은 물체에 대해 접촉점 2개 vs 3개의 wrench 공간 차이
- **form closure vs force closure 비교** (추가)
  - form closure 성립하지만 force closure 안 되는 예제
  - 마찰 추가 시 force closure로 전환되는 과정
