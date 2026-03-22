"""
그래스프 분석 모듈 (ch05 자코비안 + ch07 폐연쇄 기구학)

물체 접촉 시:
  1. 그래스프 행렬 G (6×3k): 접촉력 → 물체 wrench
  2. Force closure 판별: G의 rank + convex hull
  3. 폐연쇄 자유도 분석
  4. 구속 자코비안: 손가락 관절 → 접촉점 속도 → 물체 속도

이론:
  - 접촉점 i에서의 접촉 wrench: w_i = [f_i; r_i × f_i] (point contact)
  - 그래스프 행렬: w_obj = G @ [f_1; f_2; ...; f_k]
  - Force closure: rank(G) = 6 이고, 양의 접촉력 조합으로 임의 wrench 생성 가능
  - 폐연쇄 DOF: Grübler 공식 m = 6(N-1-J) + Σf_i
"""

import numpy as np
import mujoco


def skew(v):
    """3-벡터 → 3×3 skew-symmetric 행렬"""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])


class GraspAnalyzer:
    """MuJoCo 접촉 데이터 기반 실시간 그래스프 분석"""

    def __init__(self, model, data, fingertip_geom_names, obj_body_name):
        self.model = model
        self.data = data

        # fingertip geom IDs
        self.fingertip_gids = []
        for name in fingertip_geom_names:
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            self.fingertip_gids.append(gid)
        self.fingertip_gid_set = set(self.fingertip_gids)

        # 물체 body ID
        self.obj_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj_body_name)

    def get_fingertip_contacts(self):
        """fingertip-물체 접촉 정보 수집

        Returns:
            contacts: list of dict {
                'finger_idx': int,
                'pos': (3,) 접촉점 위치 (world),
                'normal': (3,) 접촉 법선 (물체 → 손가락 방향),
                'force': (3,) 접촉력 (world),
                'force_norm': float,
            }
        """
        contacts = []
        for c_idx in range(self.data.ncon):
            contact = self.data.contact[c_idx]

            # fingertip geom 확인
            finger_idx = None
            is_geom1_finger = contact.geom1 in self.fingertip_gid_set
            is_geom2_finger = contact.geom2 in self.fingertip_gid_set

            if not (is_geom1_finger or is_geom2_finger):
                continue

            # 물체와의 접촉인지 확인
            body1 = self.model.geom_bodyid[contact.geom1]
            body2 = self.model.geom_bodyid[contact.geom2]
            if body1 != self.obj_bid and body2 != self.obj_bid:
                continue

            if is_geom1_finger:
                finger_idx = self.fingertip_gids.index(contact.geom1)
            else:
                finger_idx = self.fingertip_gids.index(contact.geom2)

            # 접촉력
            f_local = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, c_idx, f_local)
            R_frame = contact.frame.reshape(3, 3)
            f_global = R_frame @ f_local[:3]
            f_norm = np.linalg.norm(f_global)

            if f_norm < 1e-6:
                continue

            # 접촉 법선 (frame의 첫 번째 행 = 법선 방향)
            normal = R_frame[0].copy()

            contacts.append({
                'finger_idx': finger_idx,
                'pos': contact.pos.copy(),
                'normal': normal,
                'force': f_global,
                'force_norm': f_norm,
            })

        return contacts

    def compute_grasp_matrix(self, contacts):
        """그래스프 행렬 G (6 × 3k) 계산

        Point contact with friction: 각 접촉점에서 3D 힘 벡터
        G = [  I_3      I_3     ...  ]   (3k columns)
            [ [r1×]    [r2×]    ...  ]

        여기서 r_i = p_contact_i - p_obj_com
        """
        if not contacts:
            return None

        obj_pos = self.data.xpos[self.obj_bid].copy()
        k = len(contacts)
        G = np.zeros((6, 3 * k))

        for i, c in enumerate(contacts):
            r = c['pos'] - obj_pos  # 물체 CoM에서 접촉점까지 벡터
            G[:3, 3*i:3*i+3] = np.eye(3)       # 힘 → 힘
            G[3:, 3*i:3*i+3] = skew(r)         # 힘 → 토크 (r × f)

        return G

    def check_force_closure(self, G, contacts):
        """Force closure 판별

        조건:
        1. rank(G) = 6 (full rank)
        2. 접촉점이 3개 이상 (3D에서)
        3. G의 최소 특이값 > 0 (실질적으로 rank 6)

        Returns:
            is_closure: bool
            rank: int
            min_sv: float (최소 특이값)
            quality: float (최소 특이값 / 최대 특이값 = isotropy index)
        """
        if G is None or len(contacts) < 2:
            return False, 0, 0.0, 0.0

        U, S, Vt = np.linalg.svd(G)
        rank = np.sum(S > 1e-6)
        min_sv = S[min(5, len(S)-1)] if len(S) > 0 else 0.0
        max_sv = S[0] if len(S) > 0 else 1.0
        quality = min_sv / max_sv if max_sv > 1e-10 else 0.0

        # Force closure: rank 6 + 최소 특이값 충분히 큼
        is_closure = (rank >= 6) and (min_sv > 1e-4)

        return is_closure, rank, min_sv, quality

    def compute_closed_chain_dof(self, contacts):
        """폐연쇄 자유도 분석 (Grübler 공식)

        m = 6(N - 1 - J) + Σ f_i

        여기서:
        - N: 링크 수 (물체 + 각 손가락 체인의 링크들)
        - J: 관절 수
        - f_i: 각 관절의 자유도

        간략화: 3-finger gripper에서
        - 물체: 1 body (6 DOF free joint → 접촉 구속으로 제한)
        - 각 손가락: 4 관절 (palm, upper, lower, fingertip) = 4 DOF
        - 접촉 구속: point contact with friction = 3 구속/접촉점

        실제 DOF = 총 관절 DOF - 접촉 구속 수
        """
        n_contacts = len(contacts)
        if n_contacts == 0:
            return {'total_joint_dof': 12, 'constraints': 0,
                    'closed_chain_dof': 12, 'object_dof': 6}

        # 그리퍼 관절 DOF (3 fingers × 4 joints = 12)
        gripper_dof = 12
        # 물체 DOF (free joint = 6)
        object_dof = 6
        # 접촉 구속 (point contact with friction = 3 per contact)
        n_constraints = 3 * n_contacts

        # 폐연쇄 DOF
        total_dof = gripper_dof + object_dof
        closed_chain_dof = max(0, total_dof - n_constraints)

        # 물체의 잔여 DOF
        # G의 rank가 물체에 가해지는 독립 wrench 수
        # 물체 잔여 DOF = 6 - rank(G에서 구속되는 방향 수)
        object_residual_dof = max(0, 6 - min(n_constraints, 6))

        return {
            'total_joint_dof': total_dof,
            'constraints': n_constraints,
            'closed_chain_dof': closed_chain_dof,
            'object_dof': object_residual_dof,
            'n_contacts': n_contacts,
        }

    def analyze(self):
        """전체 그래스프 분석 실행

        Returns:
            result: dict with all analysis results
        """
        contacts = self.get_fingertip_contacts()
        G = self.compute_grasp_matrix(contacts)
        is_closure, rank, min_sv, quality = self.check_force_closure(G, contacts)
        dof_info = self.compute_closed_chain_dof(contacts)

        return {
            'n_contacts': len(contacts),
            'contacts': contacts,
            'G': G,
            'force_closure': is_closure,
            'G_rank': rank,
            'min_singular_value': min_sv,
            'grasp_quality': quality,  # isotropy index
            **dof_info,
        }
