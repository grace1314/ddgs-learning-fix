# ddgs-learning-fix (`ddgs-learning&fix`)

围绕稀疏视角三维高斯重建项目 **D²GS** 的源码深度拆解、Bug 修复、**针状毛刺高斯（Needle Gaussians）消除管线**与**自定义 CUDA 算子加速**实践仓库。

---

## 核心文档导航

- **源码拆解、踩坑记录与实验笔记**：[docs/COPILOT_STUDY_AND_FIX_NOTES.md](./docs/COPILOT_STUDY_AND_FIX_NOTES.md)
- **后续学习路线与算法改进计划（Roadmap）**：[docs/LEARNING_AND_IMPROVEMENT_ROADMAP.md](./docs/LEARNING_AND_IMPROVEMENT_ROADMAP.md)

---

## 本仓库新增改进与修复

### 1. 针状/细长条毛刺高斯根治管线（Methods 1 + 2 + 3）
针对三维高斯在自采实拍场景（尤其是弱纹理背景或视角边缘）极易长出“一长两短”细针状伪影（$s_{\max} \gg s_{\text{mid}} \ge s_{\min}$），同时严格保护物体表面健康的“扁平圆盘高斯”（$s_{\max} \approx s_{\text{mid}} \gg s_{\min}$），实现了三阶段联合优化：

- **做法 1 —— 对数空间单向解耦各向异性正则损失**（[`train.py`](./train.py)、[`arguments/__init__.py`](./arguments/__init__.py)）
  - 新增 `--lambda_anisotropy` 与 `--max_needle_ratio` 参数；
  - 在对数尺度对中间轴 `log_s_mid` 做 `.detach()` 阻断反向膨胀梯度，仅定向压缩超标最长轴 `log_s_max`，避免扁平表面圆盘被误伤或吹大导致显存溢出。
- **做法 2 —— 沿主轴定向二分分裂与大尺度/极端尖刺剪枝**（[`scene/gaussian_model.py`](./scene/gaussian_model.py)）
  - 改进 `densify_and_split`：对针状高斯（`--split_needle_ratio`）仅将最长轴除以 `2`（而非原版三轴同除 `1.6`），切分一次长宽比立即减半；
  - 改进 `densify_and_prune`：支持按 `--prune_needle_ratio` 与 `--max_ws_scale_ratio` 清理极端毛刺与失控巨型高斯；
  - 修复原版 `GaussianModel.load_ply()` 未初始化 `max_radii2D` 导致加载 `.ply` 后调用剪枝报设备不一致（CPU/CUDA）的 Bug。
- **做法 3 —— 训练后几何精滤与固定拓扑愈合微调工具**（[`tools/clean_gaussians.py`](./tools/clean_gaussians.py)）
  - 独立可复用的后处理工具：支持按针状比、世界尺度比例、最低不透明度、3D ROI 分位框与 KNN 孤立点统计滤除残余噪点，并冻结拓扑微调 500~1000 步无缝填补删点空隙，导出干净的 `point_cloud_clean.ply`。

**自采 11 视角全分辨率实拍数据集实测效果**：

| 指标 | 原始代码训练结果 | **本仓库改进后（做法 1+2+3）** |
| :--- | :---: | :---: |
| 总高斯数 | 235,074 | **222,751** |
| 细长针状毛刺 (`s_max / s_mid > 10`) | 26,178 根 | **0 根** |
| 极端长刺 (`s_max / s_mid > 20`) | 9,499 根 | **0 根** |
| 全场最大针状比 `max(s_max / s_mid)` | 1,495,579.88 | **8.00（严格受控）** |
| **PSNR ($\uparrow$)** | 35.08 dB | **36.36 dB (+1.28 dB)** |
| **SSIM ($\uparrow$)** | 0.9626 | **0.9646** |
| **LPIPS ($\downarrow$)** | 0.1736 | **0.1678** |

---

### 2. 纯 GPU 莫顿码包围盒 6-NN 密度算子加速（最高 27.7×）
原版 `GaussianModel.update_density_score()` 每 500 步将点云从 GPU 拷回 CPU 运行单线程 `sklearn.neighbors.NearestNeighbors(n_neighbors=6)`，在大规模点云下造成严重停顿。

- 在 [`submodules/simple-knn/`](./submodules/simple-knn) 中基于 3D Morton Z-order 基数排序与 1024 点 AABB 包围盒裁剪架构，新增模板核函数 `boxMeanL2DistK<6>` 与 PyBind11 接口 **`meanDistCUDA6`**（同时修复了原版包围盒归约初值 `{0,0,0}` 在全正/全负坐标区间下的边界隐患）。
- 全程零 CPU-GPU 拷贝，复杂度 $O(N \log N)$，数值与 `sklearn` 完全一致。

**RTX PRO 3000 Blackwell (12GB) 实测耗时对比**：

| 高斯点云规模 | 原版 CPU `sklearn` | GPU `PyKeOps` ($O(N^2)$) | **本仓库 CUDA `meanDistCUDA6`** | 相对原版加速比 |
| :--- | :---: | :---: | :---: | :---: |
| 63,155 点 (`LLFF trex`) | 158.18 ms | 9.81 ms | **14.01 ms** | **11.3×** |
| 235,074 点 (实拍笔记本) | 594.48 ms | 106.74 ms | **32.69 ms** | **18.2×** |
| 600,000 点 (大场景规模) | 3,136.74 ms | 721.19 ms | **113.34 ms** | **27.7×** |

---

## 环境安装（适配 WSL2 + Ubuntu 24.04 & 新一代 Blackwell `sm_120` 显卡）

```bash
git clone https://github.com/grace1314/ddgs-learning-fix.git
cd ddgs-learning-fix

conda create -n ddgs python=3.10 -y
conda activate ddgs

# 安装支持 CUDA 12.8 (含 sm_120 Blackwell 架构) 的 PyTorch
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install "numpy<2" plyfile tqdm matplotlib "torchmetrics==1.2.0" opencv-python scipy scikit-learn "pykeops==2.3.0" tensorboard

# 编译两个 CUDA 核心扩展（GCC 13 环境下显式引入 cstdint / cfloat）
export CXXFLAGS="-include cstdint -include cfloat"
export NVCC_PREPEND_FLAGS="-include cstdint -include cfloat"
pip install --no-build-isolation ./submodules/simple-knn
pip install --no-build-isolation ./submodules/diff-gaussian-rasterization
```

---

## 快速上手指南

### 1. 标准稀疏视角训练与评测（以 LLFF `trex` 3 视角为例）
```bash
python train.py -s dataset/nerf_llff_data/trex -m output_llff/trex/run_1 \
    --depth_weight 0.15 --density_weight 0.85 --drop_min 0.05 --drop_max 0.5 \
    --mask_param 5 --lambda_far 0.5 --eval -r 8 --n_views 3

python render.py -m output_llff/trex/run_1 --skip_train --eval -r 8
```

### 2. 启用无毛刺增强训练 + 后处理愈合精修（适合自采全视角/实拍物体重建）
```bash
# 阶段一：开启针状各向异性约束 + 主轴定向分裂与剪枝训练
python train.py -s dataset/your_scene -m output_your_scene \
    --depth_weight 0.15 --density_weight 0.85 --drop_min 0.05 --drop_max 0.4 \
    --lambda_anisotropy 0.05 --max_needle_ratio 8.0 \
    --split_needle_ratio 12.0 --prune_needle_ratio 16.0 --max_ws_scale_ratio 0.5 \
    --densify_until_iter 8500 -r 1 --n_views 11

# 阶段二：几何精滤 + 1000 步固定拓扑表面愈合微调（生成 point_cloud_clean.ply）
python tools/clean_gaussians.py -m output_your_scene \
    --iteration 10000 --clean_needle_ratio 8.0 --clean_ws_scale_ratio 0.5 \
    --clean_min_opacity 0.02 --finetune_iters 1000

# 渲染评测精修后的模型
python render.py -m output_your_scene --iteration 11000 --skip_test -r 1
```

### 3. 跨随机种子几何稳定性评测（MW2 / IMR）
```bash
python Robustness_Assessment.py --folder output_llff/trex --device cuda
```

---

## 上游项目与参考链接

本仓库基于以下开源项目进行学习研究与功能改进，感谢原作者们的开源贡献：

- **D²GS 官方仓库（上游基线）**：[Insta360-Research-Team/DDGS](https://github.com/Insta360-Research-Team/DDGS)（[Project Page](https://insta360-research-team.github.io/DDGS-website/)）
- **DropGaussian**：[DCVL-3D/DropGaussian_release](https://github.com/DCVL-3D/DropGaussian_release)
- **3D Gaussian Splatting**：[graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)

