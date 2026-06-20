#!/bin/bash
set -e  # 任何命令失败立即退出

echo "===== 安装 CPU 版 PyTorch ====="
pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cpu

echo "===== 安装 ultralytics ====="
pip install --no-cache-dir ultralytics

echo "===== 安装 opencv ====="
pip install --no-cache-dir opencv-python-headless

echo "===== 安装完成 ====="
