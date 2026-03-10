# Chapter 9: 궤적 생성 (Trajectory Generation)

## 구현 목록

### `point_to_point.py` — 점대점 궤적
- `cubic_time_scaling(t, T)`: 3차 다항식 s(t)
- `quintic_time_scaling(t, T)`: 5차 다항식 s(t), s'(t), s''(t)
- `trapezoidal_time_scaling(t, T, v_max)`: 사다리꼴 속도 프로파일

### `joint_trajectory.py` — 관절 공간 궤적
- `joint_trajectory(q_start, q_end, T, N, method)`: 관절 보간
  - method: 'cubic', 'quintic', 'trapezoidal'
- `via_point_trajectory(q_list, T_list, N)`: 경유점 궤적
- `multi_joint_sync(q_start, q_end, dq_max, ddq_max)`: 다관절 동기화 (추가)
  - 가장 느린 관절에 맞춰 나머지 관절 시간 스케일링
  - 모든 관절이 동시에 출발/도착

### `cartesian_trajectory.py` — 작업 공간 궤적
- `screw_trajectory(T_start, T_end, T, N)`: 스크류 보간
- `cartesian_trajectory(T_start, T_end, T, N)`: 직선 보간 (R 디커플링)

### `time_optimal.py` — 시간 최적 궤적 (추가)
- `time_optimal_trajectory(robot, q_start, q_end, tau_max)`: 토크 제한 하 최단 시간 궤적
- `phase_plane_analysis(robot, q_path, tau_max)`: (s, ṡ) 위상 평면 분석
  - 최대 가속 곡선 / 최대 감속 곡선 계산
  - 스위칭 포인트 탐색
- `bang_bang_trajectory(q_start, q_end, ddq_max)`: 최대 가감속 궤적 (단순화 버전)

### `trajectory_viz.py` — 시각화
- q(t), dq(t), ddq(t) 플롯
- 3D 작업 공간 경로 애니메이션

### Pinocchio 검증
- 생성된 궤적의 각 waypoint에서 FK → 작업 공간 경로 확인

## 시각화

### ★ 가치 높음
- **q(t), dq(t), ddq(t) 시계열 3단 플롯**
  - cubic / quintic / trapezoidal 세 방법을 같은 그래프에 겹쳐서 비교
  - 가속도 불연속 (cubic) vs 연속 (quintic) 차이가 한눈에 보임
  - 관절별 색상 구분
- **3D 작업 공간 경로 애니메이션**
  - 궤적을 따라 로봇이 움직이는 애니메이션
  - EE 경로를 선으로 남기면서 진행
  - 스크류 보간 vs 직선 보간 경로 차이 비교

### ☆ 가치 보통
- **경유점 궤적 연속성**
  - 여러 경유점을 지나는 궤적에서 속도/가속도 연속성 확인
  - 경유점 위치에 마커, 속도 프로파일에 경유점 시각 표시
- **시간 스케일링 s(t) 비교**
  - s(t), ṡ(t), s̈(t)를 세 방법 나란히 플롯
  - 사다리꼴의 가감속 구간이 명확히 보이는 그래프
- **(s, ṡ) 위상 평면** (추가)
  - 시간 최적 궤적의 위상 평면 분석 결과
  - 최대 가속/감속 곡선 + 실제 궤적 경로
  - 토크 제한이 궤적 형태에 미치는 영향
