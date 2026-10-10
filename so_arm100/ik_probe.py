import numpy as np, mujoco
m=mujoco.MjModel.from_xml_path('mujoco_menagerie/trs_so_arm100/grasp_scene.xml'); d=mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m,d,0); mujoco.mj_forward(m,d)
sid=m.site('tcp').id; bid=m.body('Fixed_Jaw').id
print('tcp home',d.site_xpos[sid]); R=d.xmat[bid].reshape(3,3); print('jaw axes x,y,z cols\n',R.round(2))
