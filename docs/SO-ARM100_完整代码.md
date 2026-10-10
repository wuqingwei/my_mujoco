## 11. 完整代码

### sim.py：关节路点动画

```python
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
```

### grasp.py：IK 抓取与搬运

```python
import numpy as np, mujoco, imageio, sys
m=mujoco.MjModel.from_xml_path('mujoco_menagerie/trs_so_arm100/grasp_scene.xml')
d=mujoco.MjData(m); dk=mujoco.MjData(m)
sid=m.site('tcp').id; bid=m.body('Fixed_Jaw').id
lo=m.jnt_range[:6,0]; hi=m.jnt_range[:6,1]
def ik(target, q0, yaw=None, iters=300):
    q=q0.copy()
    for _ in range(iters):
        dk.qpos[:6]=q; mujoco.mj_kinematics(m,dk); mujoco.mj_comPos(m,dk)
        p=dk.site_xpos[sid]; R=dk.xmat[bid].reshape(3,3)
        ep=target-p
        er=np.cross(R[:,1],[0,0,1])           # y axis up -> fingers down
        Jp=np.zeros((3,m.nv)); Jr=np.zeros((3,m.nv))
        mujoco.mj_jacSite(m,dk,Jp,Jr,sid)
        J=np.vstack([Jp[:,:5],0.3*Jr[:,:5]]); e=np.r_[ep,0.3*er]
        rows=[J]; es=[e]
        if yaw is not None:  # jaw x axis horizontal direction
            ex=np.cross(R[:,0],[np.cos(yaw),np.sin(yaw),0])
            rows.append(0.2*Jr[2:3,:5]); es.append([0.2*ex[2]])
        J=np.vstack(rows); e=np.concatenate(es)
        dq=J.T@np.linalg.solve(J@J.T+1e-4*np.eye(len(e)),e)
        q[:5]=np.clip(q[:5]+dq,lo[:5],hi[:5])
        if np.linalg.norm(ep)<5e-4 and np.linalg.norm(er)<1e-2: break
    return q, np.linalg.norm(ep), np.linalg.norm(er)
home=m.key_qpos[0][:6].copy(); rest=m.key_qpos[1][:6].copy()
A=np.array([0.12,-0.2]); B=np.array([-0.12,-0.2])
OPEN, CLOSE = 1.0, -0.1
def pose(xy,z,q0,jaw):
    q,ep,er=ik(np.r_[xy,z],q0); q[5]=jaw; print('ik',xy,z,round(ep*1000,2),'mm',round(er,3)); return q
qa_hi=pose(A,0.09,home,OPEN); qa_lo=pose(A,0.012,qa_hi,OPEN)
# align cube yaw with gripper closing axis
dk.qpos[:6]=qa_lo; mujoco.mj_kinematics(m,dk); xa=dk.xmat[bid].reshape(3,3)[:,0]
yaw=np.arctan2(xa[1],xa[0])
qa_cl=qa_lo.copy(); qa_cl[5]=CLOSE
qa_up=pose(A,0.09,qa_lo,CLOSE)
qb_hi=pose(B,0.09,qa_up,CLOSE); qb_lo=pose(B,0.015,qb_hi,CLOSE)
qb_op=qb_lo.copy(); qb_op[5]=OPEN
qb_up=pose(B,0.09,qb_lo,OPEN)
seq=[(rest,0),(home,1.5),(qa_hi,1.5),(qa_lo,1.2),(qa_cl,0.8),(qa_cl,0.4),(qa_up,1.2),(qb_hi,1.6),(qb_lo,1.2),(qb_op,0.6),(qb_op,0.3),(qb_up,1.0),(home,1.2),(rest,1.5)]
mujoco.mj_resetDataKeyframe(m,d,1)
cq=m.jnt_qposadr[m.joint('cube').id]
d.qpos[cq+3:cq+7]=[np.cos(yaw/2),0,0,np.sin(yaw/2)]
mujoco.mj_forward(m,d)
r=mujoco.Renderer(m,720,960)
cam=mujoco.MjvCamera(); cam.lookat[:]=[0,-0.17,0.06]; cam.distance=0.55; cam.azimuth=110; cam.elevation=-28
fps=30; frames=[]; spf=int(round(1/fps/m.opt.timestep)); log=[]
for (a,_),(b,T) in zip(seq[:-1],seq[1:]):
    n=max(1,int(T*fps))
    for i in range(n):
        s=0.5-0.5*np.cos(np.pi*(i+1)/n); d.ctrl[:]=a+(b-a)*s
        for _ in range(spf): mujoco.mj_step(m,d)
        cam.azimuth+=0.12
        r.update_scene(d,cam); frames.append(r.render())
    log.append(d.qpos[cq:cq+3].round(3).tolist())
print('cube pos after each segment:'); [print(x) for x in log]
imageio.mimsave('so_arm100_grasp.mp4',frames,fps=fps,quality=8)
for k,f in [('g1',frames[len(frames)*45//100]),('g2',frames[len(frames)*6//10])]: imageio.imwrite(k+'.png',f)
```

### so_arm100_tcp.xml 的改动（在 Fixed_Jaw 里、fixed_jaw_pad_1 前加一行）

```xml
<site name="tcp" pos="0.0 -0.088 0" size="0.003" rgba="1 0 0 1" group="1"/>
```

### grasp_scene.xml

```xml
<mujoco model="so_arm100 grasp">
  <include file="so_arm100_tcp.xml"/>
  <option timestep="0.002"/>
  <visual><headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/><global azimuth="45" elevation="-20" offwidth="960" offheight="720"/></visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
    <geom name="padA" type="box" size="0.03 0.03 0.001" pos="0.12 -0.2 0.001" rgba="0.2 0.8 0.3 1" contype="0" conaffinity="0"/>
    <geom name="padB" type="box" size="0.03 0.03 0.001" pos="-0.12 -0.2 0.001" rgba="0.9 0.6 0.1 1" contype="0" conaffinity="0"/>
    <body name="cube" pos="0.12 -0.2 0.0125">
      <freejoint name="cube"/>
      <geom name="cube" type="box" size="0.0125 0.0125 0.0125" mass="0.015" rgba="0.9 0.15 0.15 1" friction="1.5 0.01 0.0001" condim="4"/>
    </body>
  </worldbody>
</mujoco>
```
