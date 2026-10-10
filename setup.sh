#!/usr/bin/env bash
# 拉取 MuJoCo Menagerie（只取用到的两个模型），并放入本仓库的自定义 XML
set -e
cd "$(dirname "$0")"
if [ ! -d mujoco_menagerie ]; then
  git clone --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie
  (cd mujoco_menagerie && git sparse-checkout add trs_so_arm100 universal_robots_ur10e \
     && git checkout 0059d4335f8156206f63a35662313385f7ad6d74)
fi
cp so_arm100/models/*.xml mujoco_menagerie/trs_so_arm100/
cp ur10e/models/*.xml     mujoco_menagerie/universal_robots_ur10e/
pip install -r requirements.txt
echo "完成。无显卡/无头环境运行前： export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa"
