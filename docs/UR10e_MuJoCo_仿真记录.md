# UR10e MuJoCo 仿真：安装与位姿序列演示

> 日期：2026-10-10 ｜ 环境：无显卡的 Linux 沙盒（Debian 13，2 核，7 GB），MuJoCo 3.15.0，OSMesa 软件渲染
> 结果：用 poses.json 里的关节角，演示 原点 → stowed_position → 位置4 → 位置3 → stowed_position

## 1. 安装 UR10e 模型

模型来自 google-deepmind/mujoco_menagerie 的 `universal_robots_ur10e`。在已有的 sparse checkout 里加一个目录就行：

```bash
cd mujoco_menagerie
git sparse-checkout add universal_robots_ur10e
```

目录内容：`ur10e.xml`（本体）、`scene.xml`（地面和灯光）、`assets/*.obj`（网格）。

模型要点：
- 6 个关节：shoulder_pan / shoulder_lift / elbow / wrist_1 / wrist_2 / wrist_3，与 UR 的 J1~J6 顺序一致；
- 执行器是 general 类型的 PD 控制（kp=5000，kd=500），力矩上限分别为 330 / 150 / 56 Nm；
- 关节范围：elbow 为 ±π，其余为 ±2π；
- 自带关键帧 home：`-1.5708 -1.5708 1.5708 -1.5708 -1.5708 0`；
- 末端有 `attachment_site`（法兰），没有夹爪。

### 渲染分辨率的坑

默认离屏缓冲只有 640×480，渲染 960×720 会报错，要求在 `<visual><global>` 里设置 offwidth / offheight。我没改原文件，而是复制出一份 `scene_hd.xml`：

```bash
sed 's#<global azimuth="120" elevation="-20"/>#<global azimuth="120" elevation="-20" offwidth="960" offheight="720"/>#' scene.xml > scene_hd.xml
```

### 环境重装（沙盒重置后）

```bash
pip install mujoco imageio imageio-ffmpeg
sudo apt-get install -y libosmesa6
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
```

## 2. 第一个演示：路点运动（ur10e.py）

从 home 出发，依次跑 4 个偏移路点再回到 home，每段 2 秒，用位置执行器做动力学仿真。结果：最大关节跟踪误差 0.1757 rad，出现在快速段，原因是自带 PD 增益偏软。

## 3. 第二个演示：按 poses.json 的位姿序列（ur10e_poses.py）

### 3.1 数据

poses.json 包含：custom_positions（位置1~4）、origin（原点）、poses（ps1、ps20261009）、safe_pose（安全位）、stowed_position（收纳位）。关节角为 J1→J6，单位 rad。

本次用到的：

| 名称 | J1 | J2 | J3 | J4 | J5 | J6 |
|---|---|---|---|---|---|---|
| 原点（起点） | 3.2281 | -1.0973 | -2.6058 | 0.0361 | -4.6712 | -2.4421 |
| stowed_position | 3.2324 | -0.6965 | -1.9694 | -2.5384 | -4.7222 | -2.3367 |
| 位置4 | 4.2301 | -1.0216 | -2.5011 | 0.3785 | -5.5154 | -2.3366 |
| 位置3 | 3.6906 | -1.2389 | -2.4255 | 0.1486 | -4.4652 | -2.4424 |

### 3.2 校验：模型和数据是否对得上

把各位置的关节角代入 MuJoCo 做正运动学，读 `attachment_site` 的位置，与 JSON 里的 tcp_pose 比较：

| 位置 | MuJoCo 法兰 xyz (m) | JSON TCP xyz (m) |
|---|---|---|
| 位置1 | (-0.373, 0.139, 0.520) | (-0.399, 0.141, 0.518) |
| 位置2 | (-0.274, 0.307, 0.821) | (-0.294, 0.331, 0.833) |
| 位置3 | (-0.444, -0.096, 0.552) | (-0.465, -0.111, 0.551) |
| 位置4 | (-0.348, -0.134, 0.611) | (-0.379, -0.140, 0.620) |
| 安全位 | (0.100, -0.174, 1.485) | (0.113, -0.177, 1.497) |

差 1~3 cm，方向一致，基座坐标系不用额外旋转。差值推测来自工具（TCP）偏移，模型本身符合 UR10e。

### 3.3 运动方式

- 运动学回放：直接写 qpos，不跑动力学，保证位姿和数据完全一致；
- 关节空间线性插值 + 余弦缓动，相当于 movej；
- 每段时长 = max(2 s, 1.5 × 最大关节变化量 / 0.8 rad/s)，到位后停 1 秒；
- 每帧检查接触：机械臂与地面、连杆之间。

画面叠加：当前状态与目标、J1~J6 实时数值、法兰 xyz，橙色点是末端轨迹。

### 3.4 结果

| 段 | 最大关节变化 (rad) | 时长 (s) |
|---|---|---|
| 原点 → stowed | 2.574 | 4.83 |
| stowed → 位置4 | 2.917 | 5.47 |
| 位置4 → 位置3 | 1.050 | 2.00 |
| 位置3 → stowed | 2.687 | 5.04 |

全程 639 帧，约 21 秒，960×720，0 次碰撞。

### 3.5 注意事项

安全位的 J5 = +1.571，stowed 的 J5 = -4.722，两者正好相差 2π（6.29 rad）。在真机上从安全位直接 movej 到 stowed，J5 会转一整圈。我第一次用安全位当起点时，这一段长达 11.8 秒，所以改成从原点出发。

## 4. 完整代码

### ur10e.py：路点运动

```python
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
```

### ur10e_poses.py：poses.json 位姿序列

```python
import json, numpy as np, mujoco, imageio
from PIL import Image, ImageDraw, ImageFont
m=mujoco.MjModel.from_xml_path('mujoco_menagerie/universal_robots_ur10e/scene_hd.xml'); d=mujoco.MjData(m)
P=json.load(open('/workspace/files/uploads/poses.json'))
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
```

运行：

```bash
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python3 ur10e_poses.py
```

## 5. 演示视频
