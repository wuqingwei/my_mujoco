# SO-ARM100 机械臂 MuJoCo 仿真：从零到 IK 抓取

> 日期：2026-10-10 ｜ 环境：无显卡的 Linux 沙盒 ｜ 结果：机械臂把方块夹起，从一个位置搬到另一个位置，全程是物理仿真

## 1. 目标

在一台没有显卡、没有显示器的 Linux 机器上，跑起开源机械臂 SO-ARM100（TheRobotStudio / LeRobot 用的那款）的物理仿真，分两步：
1. 让机械臂按关节路点运动，渲染成视频；
2. 场景里放一个方块，用逆运动学（IK）算出抓取姿态，真的把方块夹起来，搬到另一个位置放下。

## 2. 环境

| 项目 | 值 |
|---|---|
| 系统 | Debian GNU/Linux 13 (trixie) |
| CPU / 内存 | 2 核 / 7 GB |
| GPU / 显示 | 无 / 无（headless） |
| Python | 3.12.15 |
| MuJoCo | 3.15.0（pip） |
| 机械臂模型 | google-deepmind/mujoco_menagerie → trs_so_arm100 |

## 3. 安装

```bash
mkdir -p armsim && cd armsim
pip install mujoco imageio imageio-ffmpeg

# 只拉 menagerie 里的 SO-ARM100 目录（sparse checkout）
git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie
cd mujoco_menagerie && git sparse-checkout set trs_so_arm100
```

目录里有 `so_arm100.xml`（机械臂本体）、`scene.xml`（带地面和灯光的场景），以及 `assets/` 下的 STL 网格。

## 4. 无头渲染的坑：EGL 不可用 → 改用 OSMesa

第一次用 `MUJOCO_GL=egl` 运行，报错：

```
AttributeError: 'NoneType' object has no attribute 'eglQueryString'
```

原因：机器上没有 GPU，也没有 EGL 驱动。解决办法是改用软件渲染 OSMesa：

```bash
sudo apt-get install -y libosmesa6
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python3 sim.py
```

渲染速度够用：640×480、450 帧的视频在几十秒内就渲完了。

## 5. 模型要点

6 个关节和对应的位置执行器（position actuator，kp=50，力矩上限 ±3.5）：

| 关节 | 轴 | 范围 (rad) |
|---|---|---|
| Rotation（底座旋转） | y | -1.92 ~ 1.92 |
| Pitch（肩） | x | -3.32 ~ 0.174 |
| Elbow（肘） | x | -0.174 ~ 3.14 |
| Wrist_Pitch | x | -1.66 ~ 1.66 |
| Wrist_Roll | y | -2.79 ~ 2.79 |
| Jaw（夹爪） | z | -0.174 ~ 1.75 |

自带两个关键帧：
- `home`：`0 -1.57 1.57 1.57 -1.57 0`（伸出，夹爪朝下）
- `rest`：`0 -3.32 3.11 1.18 0 -0.174`（收拢）

夹爪上已有摩擦垫碰撞体（`fixed_jaw_pad_*`、`moving_jaw_pad_*`），可以真实接触抓取。

## 6. 第一步：关节路点动画（sim.py）

思路：定义一串关节角路点（rest → home → 右侧张爪 → 下探 → 合爪 → 抬起 → 转到左侧 → 放下 → home → rest），每两个路点之间用余弦缓动插值，写进 `d.ctrl`，每帧步进 `1/fps / timestep` 次物理，再用 `mujoco.Renderer` 渲染。

```python
m = mujoco.MjModel.from_xml_path('mujoco_menagerie/trs_so_arm100/scene.xml')
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 1)          # 从 rest 开始
r = mujoco.Renderer(m, 480, 640)
for a, b in zip(wps[:-1], wps[1:]):
    for i in range(N):
        s = 0.5 - 0.5*np.cos(np.pi*(i+1)/N)    # 余弦缓动
        d.ctrl[:] = a + (b-a)*s
        for _ in range(steps_per_frame): mujoco.mj_step(m, d)
        r.update_scene(d, cam); frames.append(r.render())
imageio.mimsave('so_arm100_demo.mp4', frames, fps=30)
```

结果：15 秒、30 fps、640×480 的视频。这一步只有关节运动，没有物体。

## 7. 第二步：放方块 + IK 抓取（grasp.py）

### 7.1 加 TCP 点

在 `Fixed_Jaw` 里、两侧夹爪垫之间加一个 site，作为工具中心点（TCP），另存为 `so_arm100_tcp.xml`：

```xml
<site name="tcp" pos="0.0 -0.088 0" size="0.003" rgba="1 0 0 1" group="1"/>
```

在 home 姿态下检查：`Fixed_Jaw` 的 y 轴朝世界 +z，说明手指（-y 方向）朝下；x 轴是夹爪闭合方向。

### 7.2 场景

`grasp_scene.xml` 用 include 引入机械臂，另外加了：
- 红色方块：边长 2.5 cm、质量 15 g、自由关节（freejoint），摩擦 1.5，condim=4；
- 绿色垫子 A：(0.12, -0.20)，方块的起始位置；
- 橙色垫子 B：(-0.12, -0.20)，放置目标（只是视觉标记，不参与碰撞）。

### 7.3 逆运动学

用阻尼最小二乘法（Damped Least Squares）解前 5 个关节，夹爪关节单独设置。误差项：
- 位置误差：TCP 到目标点；
- 姿态误差：`cross(Fixed_Jaw 的 y 轴, [0,0,1])`，让手指保持竖直向下（权重 0.3）。

```python
def ik(target, q0, iters=300):
    q = q0.copy()
    for _ in range(iters):
        dk.qpos[:6] = q; mujoco.mj_kinematics(m, dk); mujoco.mj_comPos(m, dk)
        p = dk.site_xpos[sid]; R = dk.xmat[bid].reshape(3, 3)
        ep = target - p
        er = np.cross(R[:, 1], [0, 0, 1])
        Jp = np.zeros((3, m.nv)); Jr = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, dk, Jp, Jr, sid)
        J = np.vstack([Jp[:, :5], 0.3*Jr[:, :5]]); e = np.r_[ep, 0.3*er]
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4*np.eye(6), e)
        q[:5] = np.clip(q[:5] + dq, lo[:5], hi[:5])
        if np.linalg.norm(ep) < 5e-4 and np.linalg.norm(er) < 1e-2: break
    return q
```

IK 只在一份独立的 `MjData`（dk）上做纯运动学计算，不影响仿真状态。抓取点解出来后，读出夹爪 x 轴在水平面内的朝向，把方块绕 z 轴转到同一朝向，让夹爪的两个面正对方块的两个侧面。

### 7.4 抓取序列

夹爪开度：张开 1.0 rad，闭合 -0.1 rad（闭合指令比方块宽度更紧，靠力矩上限夹持）。

| 段 | 动作 | 时长 (s) |
|---|---|---|
| 1 | rest → home | 1.5 |
| 2 | 移到 A 上方 9 cm，张爪 | 1.5 |
| 3 | 下降到 z=1.2 cm | 1.2 |
| 4 | 合爪，保持 | 0.8 + 0.4 |
| 5 | 抬起到 9 cm | 1.2 |
| 6 | 平移到 B 上方 | 1.6 |
| 7 | 下降到 z=1.5 cm | 1.2 |
| 8 | 张爪，保持 | 0.6 + 0.3 |
| 9 | 抬起 → home → rest | 1.0 + 1.2 + 1.5 |

### 7.5 结果

IK 残差：所有姿态 ≤ 0.32 mm，朝向误差约 0。

方块位置（每段结束时记录）：

| 时刻 | 方块位置 (x, y, z) m |
|---|---|
| 初始 | (0.120, -0.200, 0.012) |
| 下探后 | (0.109, -0.203, 0.012)，被夹爪顶偏约 1 cm |
| 抬起后 | (0.114, -0.196, 0.078)，离地约 6.6 cm |
| 到 B 上方 | (-0.121, -0.200, 0.080) |
| 放下后 | (-0.124, -0.198, 0.012)，离目标中心约 4 mm |

视频：960×720、30 fps、约 14 秒。

## 8. 遇到的问题和心得

1. **无 GPU 渲染**：EGL 不可用时，直接装 `libosmesa6`，用 `MUJOCO_GL=osmesa`。
2. **TCP 定义**：menagerie 的模型没有末端 site，需要自己在夹爪垫中间加一个，IK 才有目标点。
3. **5 自由度的约束分配**：机械臂只有 5 个关节决定末端位姿（第 6 个是夹爪），没法约束完整 6D 位姿。所以只约束位置 + 手指朝下，绕竖直轴的转角交给结果，再把方块转到匹配的朝向。
4. **下探时会顶到方块**：张开的夹爪下降时碰到方块侧面，方块被推偏约 1 cm。改进方法：张得更大，下探点抬高一点，或者先停在方块正上方再慢慢下降。
5. **夹持力来自位置控制**：闭合目标比方块更小，执行器顶在力矩上限（±3.5），方块靠摩擦夹住。

## 9. 下一步

- 接入 ROS 2 + MoveIt（用 RoboStack/conda 安装，Debian 13 没有官方二进制），用规划器自动避障；
- 用相机渲染图做视觉定位，不再写死方块坐标；
- 接到 LeRobot，用仿真数据做模仿学习。

## 10. 文件清单

- `sim.py`：关节路点动画
- `grasp.py`：IK 抓取与搬运
- `mujoco_menagerie/trs_so_arm100/so_arm100_tcp.xml`：加了 TCP site 的模型
- `mujoco_menagerie/trs_so_arm100/grasp_scene.xml`：抓取场景
- `so_arm100_demo.mp4`、`so_arm100_grasp.mp4`：结果视频
