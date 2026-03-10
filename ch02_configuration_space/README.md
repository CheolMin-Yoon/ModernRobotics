# Chapter 2: 상태 공간 (Configuration Space)

## 구현 목록

### `config_space.py` — 상태 공간 기초
- 자유도 계산: `gruebler(n_links, n_joints, joint_dofs)`
- 2D/3D 로봇의 C-space 시각화
  - 2-link planar: (θ1, θ2) 공간에서 장애물 매핑
  - 3DOF RRR: 관절 제한 영역 시각화

### `topology.py` — 위상 표현
- S¹ (원), T² (토러스) 시각화
- 관절 공간의 위상 구조 설명 + 그림

## 시각화

### ★ 가치 높음
- **C-space 장애물 매핑** (2-link planar)
  - (θ1, θ2) 2D 평면에 작업 공간 장애물을 C-space로 변환하여 표시
  - 자유 영역 / 장애물 영역 / 경계 색상 구분
  - 로봇 configuration 점을 C-space 위에 표시하고, 해당 자세를 작업 공간에 동시 표시 (side-by-side)

### ☆ 가치 보통
- **위상 공간 토러스**
  - 2-revolute 관절의 C-space가 T² (토러스)임을 3D 토러스 위에 경로로 시각화
  - S¹ 원 위의 점 → 관절각 대응 시각화
- **자유도 다이어그램**
  - 다양한 메커니즘 (4-bar linkage, slider-crank 등)의 자유도 계산 결과를 그림으로 표현
