#!/bin/bash
# Streamlit Cloud 部署前置脚本
# 如果 requirements.txt 中的 --extra-index-url 未生效，此脚本作为备选

pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cpu
pip install --no-cache-dir ultralytics
