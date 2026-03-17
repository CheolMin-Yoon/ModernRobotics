import mujoco
import os

urdf_path = os.path.join(os.path.dirname(__file__), '..', 'common', '3_dof_manipulator.urdf')
model = mujoco.MjModel.from_xml_path(urdf_path)
print(f"loaded: nq={model.nq}, nv={model.nv}, nbody={model.nbody}")
