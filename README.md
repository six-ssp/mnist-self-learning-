


# 🚀 PyTorch CNN 手写数字识别项目

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/) 
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

这是一个基于 **PyTorch** 深度学习框架实现的卷积神经网络（CNN）项目。

项目目标是实现对经典的 **MNIST 手写数字数据集** (0-9) 的高精度识别，并跑通深度学习项目的完整流程。


## ✨ 项目亮点

* **模型结构**: 实现了包含两层卷积、两层池化和两层全连接层的轻量级 `SimpleCNN` 结构。
* **训练加速**: 支持 **CUDA/GPU**  硬件加速训练。
* **工程化**: 代码结构清晰，将数据、模型、训练、预测逻辑分离。
* **可视化**: 预测阶段使用 `matplotlib` 库实时展示图片和预测结果。

---

## 🛠️ 环境搭建与运行

### 1. 环境准备 (Conda)

本项目基于 Conda 环境隔离。请创建并激活环境：

```bash
# 创建环境
conda create -n pytorch_mnist python=3.9
conda activate pytorch_mnist

# 安装 PyTorch 和核心依赖
# (请根据你的 CUDA 版本选择合适的 PyTorch 安装命令)
pip install torch torchvision matplotlib tqdm
````

### 2\. 模型训练

运行训练脚本。模型文件 (如 `model_epoch_10_acc_0.9880.pth`) 将自动保存在 `saved_models/` 目录下。

```bash
python src/train.py
```

### 3\. 结果预测 (推理)

运行预测脚本，随机抽取测试集图片并进行识别。程序将自动弹出图片窗口，显示识别结果和 AI 预测指数。

```bash
python src/predict.py
```

-----

## 📜 许可证

本项目采用 MIT 许可证开源。


