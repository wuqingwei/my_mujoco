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
