#!/bin/bash
# 运行贝叶斯分析脚本（解决 PyTensor 编译问题）

echo "清理 PyTensor 缓存..."
rm -rf ~/.pytensor

echo "设置编译器环境..."
export PYTENSOR_FLAGS='optimizer=fast_compile,floatX=float32,device=cpu,cxx='

echo "运行贝叶斯分析..."
python src/bayes_analysis.py