import os, numpy as np, mujoco, imageio
m = mujoco.MjModel.from_xml_path('mujoco_menagerie/trs_so_arm100/scene.xml')
d = mujoco.MjData(m)
rest = m.key_qpos[1].copy(); home = m.key_qpos[0].copy()
wps = [rest, home,
  np.array([0.8,-0.6,1.0,1.2,-1.57,1.2]),   # reach right, open jaw
  np.array([0.8,-0.3,0.9,1.3,-1.57,1.2]),   # lower
  np.array([0.8,-0.3,0.9,1.3,-1.57,0.0]),   # close jaw
  np.array([0.8,-1.2,1.3,1.4,-1.57,0.0]),   # lift
  np.array([-0.8,-1.2,1.3,1.4,0.0,0.0]),    # swing left, roll wrist
  np.array([-0.8,-0.3,0.9,1.3,0.0,0.0]),    # lower
  np.array([-0.8,-0.3,0.9,1.3,0.0,1.2]),    # open
  home, rest]
mujoco.mj_resetDataKeyframe(m, d, 1)
r = mujoco.Renderer(m, 480, 640)
cam = mujoco.MjvCamera(); cam.lookat[:] = [0,-0.1,0.12]; cam.distance=0.6; cam.azimuth=135; cam.elevation=-25
fps=30; frames=[]; seg=1.5
for a,b in zip(wps[:-1],wps[1:]):
    for i in range(int(seg*fps)):
        t=(i+1)/(seg*fps); s=0.5-0.5*np.cos(np.pi*t)
        d.ctrl[:] = a+(b-a)*s
        for _ in range(int(1/fps/m.opt.timestep)): mujoco.mj_step(m,d)
        cam.azimuth += 0.15
        r.update_scene(d, cam); frames.append(r.render())
imageio.mimsave('so_arm100_demo.mp4', frames, fps=fps, quality=8)
imageio.imwrite('frame.png', frames[len(frames)//2])
print(len(frames), m.opt.timestep)
