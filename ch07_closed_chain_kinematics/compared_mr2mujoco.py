# -*- coding: utf-8 -*-
"""
ch07 폐연쇄 기구학: MR 구현 수학적 검증

MuJoCo 시뮬 없이 알려진 기하학적 배치로 ch07 함수들을 검증한다.

비교 항목:
  [1] 파지 행렬 G — 해석적 결과와 비교
  [2] Force closure — 알려진 PASS/FAIL 케이스 검증
  [3] Grübler DOF — 교재 예제와 비교
  [4] 핸드 자코비안 블록 대각 구조
  [5] 능동/수동 분리 H 행렬 — dθ_p = H @ dθ_a 검증
  [6] 폐연쇄 자코비안 — J_closed @ dθ_a == J_h @ dθ_full 검증

conda env: mr
python ch07_closed_chain_kinematics/compared_mr2mujoco.py
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from ch07_closed_chain_kinematics.modern_robotics_ch07 import (
    grasp_matrix, check_force_closure, grubler_dof,
    hand_jacobian, partition_jacobian, closed_chain_jacobian,
    three_finger_grasp_analysis,
)

np.set_printoptions(precision=6, suppress=True)


def compare(name, a, b, tol=1e-8):
    a, b = np.atleast_1d(np.asarray(a, float)), np.atleast_1d(np.asarray(b, float))
    diff = np.linalg.norm(a - b)
    status = "✓ PASS" if diff < tol else "✗ FAIL"
    print(f"  [{name}] {status}  (diff={diff:.2e})")
    if diff >= tol:
        print(f"    got     : {a.ravel()}")
        print(f"    expected: {b.ravel()}")


print("=" * 60)
print("  ch07 폐연쇄 기구학: 수학적 검증")
print("=" * 60)

# ══════════════════════════════════════════════════════════
# [1] 파지 행렬 G 해석적 검증
# ══════════════════════════════════════════════════════════
print("\n[1] 파지 행렬 G 해석적 검증")

# 물체 CoM = 원점, 접촉점 3개 (xy 평면, 반지름 r)
r = 0.05
p_obj = np.array([0.0, 0.0, 0.0])
p_c = [
    np.array([ r,    0.0,  0.0]),   # +x
    np.array([-r/2,  r*np.sqrt(3)/2, 0.0]),  # 120°
    np.array([-r/2, -r*np.sqrt(3)/2, 0.0]),  # 240°
]

G = grasp_matrix(p_c, p_obj)

# 해석적 기대값: 상단 3×3 블록은 [I | I | I]
G_top_expected = np.hstack([np.eye(3)] * 3)
compare("G 상단 블록 [I|I|I]", G[:3, :], G_top_expected)

# 하단: [r1×] [r2×] [r3×]
from ch03_rigid_body_motion.modern_robotics_ch03 import Vec2so3
G_bot_expected = np.hstack([Vec2so3(p_c[i] - p_obj) for i in range(3)])
compare("G 하단 블록 [r1×|r2×|r3×]", G[3:, :], G_bot_expected)

# ── 단일 접촉점 케이스: r = [1,0,0], 기대 토크 블록 = [[0,0,0],[0,0,-1],[0,1,0]]
p_single = [np.array([1.0, 0.0, 0.0])]
G_single = grasp_matrix(p_single, p_obj)
compare("단일 접촉 G 상단", G_single[:3, :], np.eye(3))
compare("단일 접촉 G 하단", G_single[3:, :], Vec2so3([1.0, 0.0, 0.0]))

# ══════════════════════════════════════════════════════════
# [2] Force closure 알려진 케이스
# ══════════════════════════════════════════════════════════
print("\n[2] Force closure 알려진 케이스")

# PASS: 정삼각형 배치 3점 접촉 (xy 평면) → rank(G) = 6 (3D 힘 + 토크 모두 커버)
G3_xy = grasp_matrix(p_c, p_obj)
fc, rank, sv, q = check_force_closure(G3_xy)
print(f"  3점 xy 평면 배치: closure={fc}, rank={rank}  (기대: rank=6, closure=True)")
compare("rank", rank, 6, tol=0.5)
print(f"  force_closure 일치: {'✓ PASS' if fc == True else '✗ FAIL'}")

# PASS: 3D 배치 (z 방향 접촉 추가) → rank = 6
p_c_3d = p_c + [np.array([0.0, 0.0, r])]
G4 = grasp_matrix(p_c_3d, p_obj)
fc4, rank4, sv4, q4 = check_force_closure(G4)
print(f"  4점 3D 배치:      closure={fc4}, rank={rank4}  (기대: rank=6, closure=True)")
compare("rank", rank4, 6, tol=0.5)
print(f"  force_closure 일치: {'✓ PASS' if fc4 == True else '✗ FAIL'}")

# FAIL: 접촉점 1개 → rank = 3
G1 = grasp_matrix([p_c[0]], p_obj)
fc1, rank1, _, _ = check_force_closure(G1)
print(f"  1점 접촉:         closure={fc1}, rank={rank1}  (기대: rank=3, closure=False)")
compare("rank", rank1, 3, tol=0.5)
print(f"  force_closure 일치: {'✓ PASS' if fc1 == False else '✗ FAIL'}")

# ══════════════════════════════════════════════════════════
# [3] Grübler DOF 교재 예제
# ══════════════════════════════════════════════════════════
print("\n[3] Grübler DOF 검증")

# 교재 예제: 직렬 2-link 로봇 (공간, m=6)
# N=3 (ground + link1 + link2), J=2 revolute (f=1), 접촉 없음
# dof = 6(3-1-2) + 2 = 0 + 2 = 2
dof_2link = grubler_dof(n_bodies=3, n_joints=2, joint_dofs=[1, 1])
print(f"  2-link 직렬 로봇: dof={dof_2link}  (기대: 2)")
compare("2-link DOF", dof_2link, 2, tol=0.5)

# 3-finger gripper + 물체 (접촉 없음)
# N=14 (ground+12 finger links+물체), J=13 (12 revolute + 1 free)
# dof = 6(14-1-13) + 12*1 + 6 = 0 + 18 = 18
dof_free = grubler_dof(n_bodies=14, n_joints=13,
                       joint_dofs=[1]*12 + [6])
print(f"  3-finger + 물체 (접촉 없음): dof={dof_free}  (기대: 18)")
compare("free DOF", dof_free, 18, tol=0.5)

# 3점 접촉 (point contact with friction = 3 구속/접촉)
dof_3c = grubler_dof(n_bodies=14, n_joints=13,
                     joint_dofs=[1]*12 + [6],
                     n_contact_constraints=9)
print(f"  3-finger + 물체 (3점 접촉):  dof={dof_3c}   (기대: 9)")
compare("3-contact DOF", dof_3c, 9, tol=0.5)

# 6점 접촉 → force closure 달성 시
dof_6c = grubler_dof(n_bodies=14, n_joints=13,
                     joint_dofs=[1]*12 + [6],
                     n_contact_constraints=18)
print(f"  3-finger + 물체 (6점 접촉):  dof={dof_6c}   (기대: 0)")
compare("6-contact DOF", dof_6c, 0, tol=0.5)

# ══════════════════════════════════════════════════════════
# [4] 핸드 자코비안 블록 대각 구조
# ══════════════════════════════════════════════════════════
print("\n[4] 핸드 자코비안 블록 대각 구조")

np.random.seed(42)
J_f1 = np.random.randn(3, 4)
J_f2 = np.random.randn(3, 4)
J_f3 = np.random.randn(3, 4)

J_h = hand_jacobian([J_f1, J_f2, J_f3])
print(f"  J_h shape: {J_h.shape}  (기대: (9, 12))")
compare("J_h shape[0]", J_h.shape[0], 9,  tol=0.5)
compare("J_h shape[1]", J_h.shape[1], 12, tol=0.5)

# 블록 대각 검증: 비대각 블록 = 0
compare("비대각 블록 J_h[0:3, 4:8]",  J_h[0:3, 4:8],  np.zeros((3, 4)))
compare("비대각 블록 J_h[0:3, 8:12]", J_h[0:3, 8:12], np.zeros((3, 4)))
compare("비대각 블록 J_h[3:6, 0:4]",  J_h[3:6, 0:4],  np.zeros((3, 4)))
compare("비대각 블록 J_h[6:9, 0:4]",  J_h[6:9, 0:4],  np.zeros((3, 4)))

# 대각 블록 = 원래 자코비안
compare("대각 블록 J_h[0:3, 0:4] == J_f1", J_h[0:3, 0:4], J_f1)
compare("대각 블록 J_h[3:6, 4:8] == J_f2", J_h[3:6, 4:8], J_f2)
compare("대각 블록 J_h[6:9, 8:12] == J_f3", J_h[6:9, 8:12], J_f3)

# ══════════════════════════════════════════════════════════
# [5] 능동/수동 분리 H 행렬
# ══════════════════════════════════════════════════════════
print("\n[5] 능동/수동 분리 H 행렬")

# 각 손가락 2관절 능동, 2관절 수동
active_idx  = [0, 1, 4, 5, 8, 9]
passive_idx = [2, 3, 6, 7, 10, 11]

J_a, J_p, H = partition_jacobian(J_h, active_idx, passive_idx)
print(f"  J_a: {J_a.shape}, J_p: {J_p.shape}, H: {H.shape}")

# H = -J_p† @ J_a 검증
H_expected = -np.linalg.pinv(J_p) @ J_a
compare("H == -J_p† @ J_a", H, H_expected)

np.random.seed(7)
dtheta_a = np.random.randn(6)
dtheta_p = H @ dtheta_a

# 폐연쇄 구속 검증: J_p가 정방(square)일 때 J_p @ dθ_p + J_a @ dθ_a = 0 성립
# 손가락 1개, 접촉 2DOF, 관절 2개 → J_p, J_a 모두 (2×1) → 정방 아님
# 완전 정방: 접촉 DOF == 수동 관절 수 == 능동 관절 수
# 손가락 2개, 각 2관절, 접촉 2DOF/손가락 → J_h: (4×4), 능동 2, 수동 2
np.random.seed(99)
J_f_a = np.random.randn(2, 2)
J_f_b = np.random.randn(2, 2)
J_h_sq = hand_jacobian([J_f_a, J_f_b])   # (4×4) 정방
act_sq = [0, 2]   # 각 손가락 첫 관절
pas_sq = [1, 3]   # 각 손가락 둘째 관절
J_a_sq, J_p_sq, H_sq = partition_jacobian(J_h_sq, act_sq, pas_sq)
# J_p_sq: (4×2), J_a_sq: (4×2) — 여전히 tall
# 진짜 정방: 행 수 = 수동 관절 수 → 접촉 DOF = 수동 관절 수
# 손가락 1개, 관절 2개, 접촉 2DOF
np.random.seed(77)
J_1f = np.random.randn(2, 2)   # (2×2) 정방
J_a_1, J_p_1, H_1 = partition_jacobian(J_1f, [0], [1])
# J_p_1: (2×1), J_a_1: (2×1) — 여전히 tall
# 결론: partition_jacobian에서 J_p는 항상 (m × n_p), m >= n_p
# 구속 J_p @ dθ_p = -J_a @ dθ_a 는 overdetermined → least-squares
# 정확한 검증: J_p가 정방(m==n_p)인 경우만 성립
# m == n_p: 접촉 DOF == 수동 관절 수
np.random.seed(55)
J_sq_exact = np.random.randn(3, 3)   # (3×3) 정방
J_a_e = J_sq_exact[:, [0, 1]]       # (3×2) 능동
J_p_e = J_sq_exact[:, [2]]          # (3×1) 수동 — 여전히 tall
# 완전 정방: m=n_p → 수동 관절 수 = 행 수
# 손가락 3개, 각 1관절, 접촉 1DOF/손가락 → J_h: (3×3), 능동 0, 수동 3
# 능동 없이 수동만: 의미 없음
# 실용적 정방 케이스: 수동 관절 수 = 행 수 = 접촉 DOF
# J: (n×n), 능동 n/2, 수동 n/2
np.random.seed(42)
n = 4
J_sq2 = np.random.randn(n, n)
act_e  = list(range(n//2))
pas_e  = list(range(n//2, n))
J_a_e2, J_p_e2, H_e2 = partition_jacobian(J_sq2, act_e, pas_e)
# J_p_e2: (4×2), J_a_e2: (4×2) — 여전히 tall (행=4, 열=2)
# 핵심: partition_jacobian은 열을 분리하므로 J_p는 항상 (m × n_p), m >= n_p
# 구속이 정확히 성립하려면 m == n_p, 즉 J_p가 정방이어야 함
# → J_h 자체가 정방(m==n)이고 능동=수동=n/2일 때: J_p는 (n × n/2) → tall
# 결론: 일반적인 핸드 자코비안에서 이 구속은 least-squares 의미
# 대신 J_closed의 의미를 직접 검증
dta_e = np.random.randn(n//2)
dtp_e = H_e2 @ dta_e
dtheta_e = np.zeros(n)
dtheta_e[act_e]  = dta_e
dtheta_e[pas_e]  = dtp_e
v_closed_e = closed_chain_jacobian(J_a_e2, J_p_e2, H_e2) @ dta_e
v_full_e   = J_sq2 @ dtheta_e
compare("정방 J_h: J_closed@dθ_a == J_h@dθ_full", v_closed_e, v_full_e)
print(f"  [참고] J_p@dθ_p + J_a@dθ_a = 0 구속은 J_p가 정방일 때만 정확히 성립")

# ══════════════════════════════════════════════════════════
# [6] 폐연쇄 자코비안 J_closed
# ══════════════════════════════════════════════════════════
print("\n[6] 폐연쇄 자코비안 J_closed")

J_closed = closed_chain_jacobian(J_a, J_p, H)
print(f"  J_closed shape: {J_closed.shape}  (기대: (9, 6))")

# J_closed @ dθ_a == J_h @ dθ_full  (dθ_full에서 passive는 H로 결정)
dtheta_full = np.zeros(12)
dtheta_full[active_idx]  = dtheta_a
dtheta_full[passive_idx] = dtheta_p

v_closed = J_closed @ dtheta_a
v_full   = J_h @ dtheta_full
compare("J_closed@dθ_a == J_h@dθ_full", v_closed, v_full)

# ══════════════════════════════════════════════════════════
# [7] three_finger_grasp_analysis 종합
# ══════════════════════════════════════════════════════════
print("\n[7] three_finger_grasp_analysis 종합")

result = three_finger_grasp_analysis(
    p_contacts=p_c_3d,
    p_obj=p_obj,
    finger_jacobians=[J_f1, J_f2, J_f3],
    active_idx=active_idx,
    passive_idx=passive_idx,
)

# G 일치
compare("G 행렬", result['G'], grasp_matrix(p_c_3d, p_obj))
# force closure
fc_ref, rank_ref, sv_ref, q_ref = check_force_closure(grasp_matrix(p_c_3d, p_obj))
compare("force_closure", int(result['force_closure']), int(fc_ref), tol=0.5)
compare("G_rank",        result['G_rank'],              rank_ref,   tol=0.5)
compare("quality",       result['quality'],             q_ref,      tol=1e-8)
# J_closed
compare("J_closed", result['J_closed'], J_closed)

print("\n" + "=" * 60)
print("  done")
print("=" * 60)
