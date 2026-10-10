import json, numpy as np, mujoco, imageio
from PIL import Image, ImageDraw, ImageFont
m=mujoco.MjModel.from_xml_path('mujoco_menagerie/universal_robots_ur10e/scene_hd.xml'); d=mujoco.MjData(m)
P=json.load(open('ur10e/poses.json'))
cp={c['name']:c['joints_rad'] for c in P['custom_positions']}
seq=[('origin (start)',P['origin']['joints_rad']),('stowed_position',P['stowed_position']),
     ('Position 4',cp['位置4']),('Position 3',cp['位置3']),('stowed_position',P['stowed_position'])]
sid=m.site('attachment_site').id; floor=m.geom('floor').id
fps=30; VMAX=0.8  # rad/s peak joint speed
r=mujoco.Renderer(m,720,960); cam=mujoco.MjvCamera()
cam.lookat[:]=[-0.15,0,0.55]; cam.distance=2.3; cam.azimuth=200; cam.elevation=-18
font=ImageFont.load_default(size=26); small=ImageFont.load_default(size=20)
frames=[]; trail=[]; contacts=0; log=[]
def frame(q,label,status):
    global contacts
    d.qpos[:6]=q; mujoco.mj_forward(m,d)
    for i in range(d.ncon):
        c=d.contact[i]
        if floor in (c.geom1,c.geom2) or m.geom_bodyid[c.geom1]!=m.geom_bodyid[c.geom2]: contacts+=1
    trail.append(d.site_xpos[sid].copy())
    r.update_scene(d,cam)
    for p in trail[::3]:
        if r.scene.ngeom>=r.scene.maxgeom: break
        mujoco.mjv_initGeom(r.scene.geoms[r.scene.ngeom],mujoco.mjtGeom.mjGEOM_SPHERE,[0.008,0,0],p,np.eye(3).flatten(),[1,0.5,0,0.8]); r.scene.ngeom+=1
    img=Image.fromarray(r.render()); dr=ImageDraw.Draw(img)
    dr.rectangle([0,0,960,96],fill=(0,0,0))
    dr.text((16,10),f'UR10e  {status}: {label}',font=font,fill=(255,255,255))
    dr.text((16,52),'J1..J6 = '+'  '.join(f'{x:+.3f}' for x in q)+'  rad',font=small,fill=(180,220,255))
    tcp=d.site_xpos[sid]; dr.text((16,690),f'flange xyz = ({tcp[0]:+.3f}, {tcp[1]:+.3f}, {tcp[2]:+.3f}) m',font=small,fill=(255,255,255))
    frames.append(np.array(img)); cam.azimuth+=0.08
for (na,a),(nb,b) in zip(seq[:-1],seq[1:]):
    a=np.array(a); b=np.array(b); dq=np.abs(b-a).max()
    T=max(2.0, 1.5*dq/VMAX*1.0); n=int(T*fps)
    for i in range(n):
        s=0.5-0.5*np.cos(np.pi*(i+1)/n); frame(a+(b-a)*s, f'{na} -> {nb}','moving')
    for i in range(int(1.0*fps)): frame(b, nb,'reached')
    log.append((nb, round(T,2), round(dq,3)))
for x in log: print(x)
print('contacts',contacts,'frames',len(frames))
imageio.mimsave('ur10e_poses.mp4',frames,fps=fps,quality=7)
for k,i in [('p1',int(len(frames)*0.3)),('p2',int(len(frames)*0.62))]: imageio.imwrite(k+'.png',frames[i])
