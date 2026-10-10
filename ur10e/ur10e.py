import numpy as np, mujoco, imageio
m=mujoco.MjModel.from_xml_path('mujoco_menagerie/universal_robots_ur10e/scene_hd.xml'); d=mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m,d,0)
home=m.key_qpos[0].copy()
wps=[home,
 home+np.array([1.2,0.5,-0.6,0.1,0,1.5]),
 home+np.array([1.2,0.2,0.2,-0.4,0.6,-1.0]),
 home+np.array([-1.0,0.6,-0.9,0.3,-0.5,1.0]),
 home+np.array([-1.0,-0.1,0.3,-0.2,0,0]),
 home]
r=mujoco.Renderer(m,720,960); cam=mujoco.MjvCamera(); cam.lookat[:]=[0,0,0.45]; cam.distance=2.0; cam.azimuth=120; cam.elevation=-20
fps=30; spf=int(round(1/fps/m.opt.timestep)); frames=[]; err=[]
for a,b in zip(wps[:-1],wps[1:]):
    n=int(2.0*fps)
    for i in range(n):
        s=0.5-0.5*np.cos(np.pi*(i+1)/n); d.ctrl[:]=a+(b-a)*s
        for _ in range(spf): mujoco.mj_step(m,d)
        err.append(np.abs(d.qpos-d.ctrl).max()); cam.azimuth+=0.2
        r.update_scene(d,cam); frames.append(r.render())
imageio.mimsave('ur10e_demo.mp4',frames,fps=fps,quality=8); imageio.imwrite('ur.png',frames[len(frames)//3])
print(m.nq,m.nu,m.opt.timestep,'max track err rad',round(max(err),4),len(frames))
