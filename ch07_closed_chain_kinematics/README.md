# Chapter 7: Kinematics of Closed Chains

폐연쇄 기구학 — Cassie 로봇 (Agility Robotics) 분석

## 구현 내용

| 파일 | 설명 |
|------|------|
| `modern_robotics_ch07.py` | 스켈레톤 (구현 예정) |
| `cassie_test.py` | MuJoCo로 Cassie 로드, 폐연쇄 구속 분석, 뷰어 스폰 |

## Cassie 폐연쇄 구조

4개의 equality connect 구속으로 루프를 닫는다:

- left-achilles-rod ↔ left-heel-spring
- left-plantar-rod ↔ left-foot
- 오른쪽도 동일하게 2개

## cassie_test.py 출력 내용

- 모델 기본 정보 (nq, nv, nbody, njnt, nu, neq)
- 조인트 목록 (free, ball, hinge)
- 등식 구속 조건 (connect 타입, body 쌍, anchor)
- 구속 위반 확인 (home config)
- 왼쪽 다리 폐연쇄 body 위치
- 액추에이터 목록

## 실행

```bash
# 모델 정보 출력
python ch07_closed_chain_kinematics/cassie_test.py

# MuJoCo 뷰어로 Cassie 스폰 (시뮬레이션)
python ch07_closed_chain_kinematics/cassie_test.py --view
```
