# Chapter 10: 동작 계획 (Motion Planning)

## 구현 목록

### `rrt.py` — RRT (Rapidly-exploring Random Tree)
- `rrt(q_start, q_goal, robot, obstacles, max_iter)`: 기본 RRT
- `rrt_connect(q_start, q_goal, ...)`: 양방향 RRT
- 관절 공간 장애물 충돌 체크

### `rrt_star.py` — RRT* 최적 RRT (추가)
- `rrt_star(q_start, q_goal, robot, obstacles, max_iter)`: 점근적 최적 RRT
- rewire 로직: 새 노드 추가 시 근처 노드 비용 재계산
- RRT vs RRT* 경로 품질 비교

### `prm.py` — PRM (Probabilistic Roadmap)
- `build_roadmap(robot, obstacles, n_samples)`: 로드맵 생성
- `query(roadmap, q_start, q_goal)`: A* 경로 탐색

### `grid_search.py` — 격자 기반 탐색 (추가)
- `a_star_grid(grid, start, goal)`: 2D 격자 A* 탐색
- `dijkstra_grid(grid, start, goal)`: Dijkstra 탐색
- 샘플링 기반 (RRT) vs 격자 기반 (A*) 비교

### `potential_field.py` — 포텐셜 필드
- `attractive_potential(q, q_goal)`: 인력장
- `repulsive_potential(q, obstacles)`: 척력장
- 로컬 미니마 문제 시각화

### `collision.py` — 충돌 검사
- 간단한 구/박스 장애물 정의
- FK → 링크 위치 → 장애물 거리 계산

### `planning_viz.py` — 시각화
- 2D C-space 장애물 맵 + 경로
- 3D 작업 공간 경로 + 로봇 애니메이션

## 시각화

### ★ 가치 높음
- **C-space 장애물 + RRT 트리 성장**
  - 2-link planar: (θ1, θ2) 평면에 장애물 영역 + RRT 트리 노드/엣지
  - 트리가 점진적으로 성장하는 애니메이션
  - 최종 경로를 굵은 선으로 하이라이트
- **작업 공간 경로 + 로봇 애니메이션**
  - 계획된 경로를 따라 로봇이 장애물을 피해 움직이는 3D 애니메이션
  - 장애물을 반투명 구/박스로 표시

### ☆ 가치 보통
- **포텐셜 필드 등고선**
  - 2D 작업 공간에서 인력 + 척력 합산 포텐셜 등고선 맵
  - gradient 방향 화살표 (quiver plot)
  - 로컬 미니마에 빠지는 경로 vs 탈출하는 경로 비교
- **PRM 로드맵**
  - 샘플 노드 + 연결 엣지를 C-space에 표시
  - 쿼리 경로를 하이라이트
  - 샘플 수에 따른 로드맵 밀도 변화
- **RRT vs RRT* 경로 품질** (추가)
  - 같은 환경에서 RRT / RRT* 경로를 겹쳐서 비교
  - iteration 수에 따른 경로 비용 수렴 그래프
