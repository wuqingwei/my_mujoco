# my_mujoco

MuJoCo 机械臂仿真练习：SO-ARM100 和 UR10e。

| 演示 | 脚本 | 说明 |
|---|---|---|
| SO-ARM100 路点运动 | `so_arm100/sim.py` | 关节空间路点 + 位置执行器动力学仿真 |
| SO-ARM100 IK 抓取 | `so_arm100/grasp.py` | 阻尼最小二乘 IK，把红色方块从绿垫搬到橙垫 |
| UR10e 路点运动 | `ur10e/ur10e.py` | 从 home 出发跑 4 个路点 |
| UR10e 位姿序列 | `ur10e/ur10e_poses.py` | 读 `ur10e/poses.json`：原点 → stowed → 位置4 → 位置3 → stowed |

## 快速开始

```bash
./setup.sh                                   # 拉 Menagerie 模型 + 装依赖
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa   # 无显卡时；有显卡可用 egl 或 glfw
python3 so_arm100/grasp.py                   # 在仓库根目录运行，视频输出到当前目录
python3 ur10e/ur10e_poses.py
```

## 目录

- `so_arm100/models/`：`so_arm100_tcp.xml`（在 Fixed_Jaw 加了 tcp site）、`grasp_scene.xml`（方块和两块垫子）
- `ur10e/models/scene_hd.xml`：离屏缓冲改为 960×720 的场景
- `ur10e/poses.json`：真机保存的关节位姿（J1→J6，rad）
- `docs/`：完整过程文档
- `media/`：演示视频（压缩版）和截图

## 注意

- 安全位与 stowed_position 的 J5 相差约 2π，真机直接 movej 会让 J5 多转一整圈。
- MuJoCo 法兰位置与 poses.json 里的 TCP 差 1–3 cm，推测是工具偏移。

机器人模型来自 [google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie)（Apache-2.0），未包含在本仓库中，由 `setup.sh` 拉取。
