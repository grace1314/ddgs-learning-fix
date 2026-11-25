# D²GS (ICLR 2026) 深度代码研读、实战踩坑与改进笔记

> 本仓库基于官方 [Insta360-Research-Team/DDGS](https://github.com/Insta360-Research-Team/DDGS) 构建，完整记录了从零搭建 **WSL2 + Ubuntu 24.04 + NVIDIA Blackwell (`sm_120`)** 开发环境、项目架构拆解、稀疏视角复现、自采真实场景重建、**细长针状高斯（Needle Gaussians）裁剪修复**、**2D 超分/AI OCR 探索** 到 **自定义 CUDA KNN 算子加速（最高 27.7×）** 的全过程。

---

## 目录

1. [项目定位与三大核心创新](#一项目定位与三大核心创新)
2. [整体架构与逐模块源码解析](#二整体架构与逐模块源码解析)
3. [WSL2 + Ubuntu 24.04 + Blackwell (`sm_120`) 环境避坑指南](#三wsl2--ubuntu-2404--blackwell-sm_120-环境避坑指南)
4. [实战一：LLFF 稀疏视角（3-View）复现与原版 3DGS 对照](#四实战一llff-稀疏视角3-view复现与原版-3dgs-对照)
5. [实战二：自定义实拍数据集（笔记本 `gc_data`）全量重建要点](#五实战二自定义实拍数据集笔记本-gc_data全量重建要点)
6. [核心改进一：彻底消除“针状细长毛刺高斯”（做法 1 + 2 + 3 联合管线）](#六核心改进一彻底消除针状细长毛刺高斯做法-1--2--3-联合管线)
7. [探索实验：低清文字细节恢复（超分 vs AI OCR 矢量重绘边界分析）](#七探索实验低清文字细节恢复超分-vs-ai-ocr-矢量重绘边界分析)
8. [核心改进二：MW2 最优传输原理解析 & 自定义 CUDA `meanDistCUDA6` 算子加速](#八核心改进二mw2-最优传输原理解析--自定义-cuda-meandistcuda6-算子加速)

---

## 一、项目定位与三大核心创新

**D²GS** 针对稀疏视角（如仅 3 张或 12 张输入图）三维高斯重建中的**近景过拟合伪影、远景欠拟合模糊、跨随机种子训练不稳定**三大痛点，在原版 3DGS 基础上引入三项关键设计：

1. **DD-Drop（Depth-and-Density Guided Dropout，深度与密度引导丢弃）**
   - 每 500 步计算每个高斯的 6 近邻空间密度分 `density_score`（越密集得分越高）；
   - 每次训练前向渲染时计算相机系 $Z$ 深度反向归一化得分 `depth_score`（越近得分越高），并按 33% / 67% 深度分位切分**近场 / 中场 / 远场**；
   - 按 `near(1.0) / mid(0.7) / far(0.3)` 阶梯衰减丢弃概率（保护远处本就稀疏的高斯），随迭代从 `drop_min` 线性升温至 `drop_max`，随机置零部分高斯的 `opacity` 抑制局部死记硬背。
2. **DAFE（Distance-Aware Fidelity Enhancement，距离感知保真度增强）**
   - 在原版 $\mathcal{L}_1 + \lambda_{\text{dssim}}(1 - \text{SSIM})$ 损失之上，利用远景区域掩码 `far_mask` 叠加 $\lambda_{\text{far}} \mathcal{L}_{1,\text{far}}$，给梯度微弱的远景区域开启 $(1 + \lambda_{\text{far}})$ 倍监督放大镜。
3. **MW2 / IMR 几何分布级训练稳定性评估**
   - 将训练出的 3DGS 模型视为高斯混合分布（GMM），用 `PyKeOps LazyTensor` 在显存零爆炸前提下通过 Sinkhorn-Knopp 算法求解两两模型间的 Wasserstein-2 距离，并汇总为不稳定性指标 $\text{IMR} = \ln(\sum d_k^2 / \sum d_k)$。

---

## 二、整体架构与逐模块源码解析

```text
DDGS/
├── tools/                  # 数据预处理与后处理：COLMAP 稀疏/稠密重建脚本 + 新增 clean_gaussians.py 几何精修工具
├── arguments/              # ModelParams / PipelineParams / OptimizationParams 命令行与配置合并
├── scene/                  # Scene 场景管理、COLMAP/Blender/Replica 读取器、Camera 投影矩阵、GaussianModel 核心类
├── gaussian_renderer/      # render() 封装（含 DD-Drop 核心 35 行实现）与 network_gui 远程查看器接口
├── submodules/
│   ├── diff-gaussian-rasterization/  # Tile-based 可微高斯光栅化 CUDA 内核（前向投影排序混合 + 反向梯度）
│   └── simple-knn/                   # 基于莫顿码与 AABB 包围盒的 CUDA KNN 算子（含新增 meanDistCUDA6）
├── utils/ & lpipsPyTorch/  # L1/SSIM 损失、PSNR、球谐函数求值与预训练 VGG LPIPS 感知指标
└── train.py / render.py / metric.py / Robustness_Assessment.py
```

### 关键源码定位表
- **稀疏视角采样逻辑**：[`scene/dataset_readers.py`](../scene/dataset_readers.py) 中 `readColmapSceneInfo` 先按 `llffhold=8` 留出测试集，再用 `np.linspace(0, len-1, n_views)` 均匀抽取训练视角，并加载 `{n_views}_views/dense/fused.ply`。
- **3D 高斯 59 维可学习参数**：[`scene/gaussian_model.py`](../scene/gaussian_model.py) 中 `_xyz(3)` + `_features_dc(3)` + `_features_rest(45)` + `_scaling(3, exp激活)` + `_rotation(4, normalize激活)` + `_opacity(1, sigmoid激活)`。
- **DD-Drop 实现位置**：[`gaussian_renderer/__init__.py`](../gaussian_renderer/__init__.py) 中 `if is_train:` 分支。
- **DAFE 与各向异性正则损失**：[`train.py`](../train.py) 主训练循环损失计算段。

---

## 三、WSL2 + Ubuntu 24.04 + Blackwell (`sm_120`) 环境避坑指南

针对新一代 **NVIDIA RTX PRO 3000 Blackwell (12GB VRAM, Compute Capability `12.0`)** 工作站的环境实测要点：

1. **切勿照搬 `environment.yaml` 的 CUDA 12.1**：
   - CUDA 12.1 不支持 Blackwell `sm_120` 架构。需安装 **CUDA Toolkit 12.8 (`cuda-compiler-12-8` + `cuda-libraries-dev-12-8` + `cuda-nvrtc-dev-12-8`)** 与 **PyTorch 2.8.0+cu128**。
2. **Ubuntu 24.04 GCC 13 编译 3DGS 子模块补丁**：
   - GCC 13 移除了部分隐式头文件包含，编译两个子模块时需传入：
     ```bash
     export TORCH_CUDA_ARCH_LIST="12.0"
     export CXXFLAGS="-include cstdint -include cfloat"
     export NVCC_PREPEND_FLAGS="-include cstdint -include cfloat"
     pip install --no-build-isolation ./submodules/simple-knn
     pip install --no-build-isolation ./submodules/diff-gaussian-rasterization
     ```
3. **WSL2 内存与图形化桌面（WSLg + XFCE4/xrdp）**：
   - 在 Windows `%USERPROFILE%\.wslconfig` 配置 `memory=20GB`（`[wsl2]` 段）与 `autoMemoryReclaim=gradual`（`[experimental]` 段）。
   - 单应用 GUI（如 `colmap gui`）可直接通过系统自带 WSLg 无缝弹窗；完整 Linux 桌面可通过 `xfce4 + xrdp`（端口 `3390`）配合 Windows `mstsc` 流畅访问。

---

## 四、实战一：LLFF 稀疏视角（3-View）复现与原版 3DGS 对照

在 `nerf_llff_data/trex` 场景（仅 3 张输入图，`-r 8` 降采样，10,000 步，单次训练仅需 ~105 秒）上的实测对比：

| 方法 | 训练视角 | 测试集 PSNR ($\uparrow$) | 测试集 SSIM ($\uparrow$) | 测试集 LPIPS ($\downarrow$) |
| :--- | :---: | :---: | :---: | :---: |
| **Vanilla 3DGS 基线** (`drop=0, lambda_far=0`) | 3 | 21.62 dB | 0.8070 | 0.1595 |
| **D²GS 完整版** (`DD-Drop + DAFE`) | 3 | **21.97 dB (+0.35)** | **0.8214 (+0.0144)** | **0.1455 (-8.8%)** |

- 多次独立运行通过 `Robustness_Assessment.py` 测得稳定性指标 **IMR = 4.1455**。

---

## 五、实战二：自定义实拍数据集（笔记本 `gc_data`）全量重建要点

将自拍图像（如 10 张横拍 + 1 张竖拍特写）送入 DDGS 全图训练（不留测试集）需满足 4 个工程条件：
1. **文件名规范化**：去除空格与特殊符号（如重命名为 `img_00.jpg` ~ `img_10.jpg`），防止 COLMAP `images.txt` 空格分隔解析错位；按横竖屏分目录提取相机内参。
2. **去畸变转针孔模型**：运行 `colmap image_undistorter` 将 `OPENCV` 畸变模型转换为 DDGS 支持的 4 参数 `PINHOLE` 模型。
3. **生成稠密点云与占位 `poses_bounds.npy`**：运行 `patch_match_stereo` + `stereo_fusion` 输出 `{N}_views/dense/fused.ply`，并生成形状为 `[N, 17]` 的 `poses_bounds.npy`。
4. **全图满分辨率训练参数**：去掉 `--eval`，显式指定 `-r 1 --n_views N`（不可使用默认的 `-r -1`）。

---

## 六、核心改进一：彻底消除“针状细长毛刺高斯”（做法 1 + 2 + 3 联合管线）

### 问题诊断
原仓库 `train.py` 将 `size_threshold = None` 写死，跳过了大尺寸高斯剪枝；在自拍物体缺纹理背景或边缘视差处极易产生“一长两短”的细针毛刺（$s_{\max} \gg s_{\text{mid}} \ge s_{\min}$）。
> **关键原则**：必须区分**健康的表面扁圆盘**（$s_{\max} \approx s_{\text{mid}} \gg s_{\min}$，占场景 >50%，不可误删！）与**真正的针状毛刺**（$s_{\max} / s_{\text{mid}} \gg 1$）。

### 三步根治方案（已集成至代码库）
1. **做法 1（训练期对数空间单向解耦各向异性 Loss）** —— [`train.py`](../train.py)
   - 计算 `torch.clamp(log_s_max - log_s_mid.detach() - log(max_needle_ratio), min=0.0).mean()`。
   - **核心避坑**：必须对 `log_s_mid` 做 `.detach()`！否则分母负梯度会把细针“吹胖”成遮挡全屏的巨型圆盘导致显存爆炸。
2. **做法 2（沿主轴定向二分分裂 + 大尺寸/极端毛刺剪枝）** —— [`scene/gaussian_model.py`](../scene/gaussian_model.py)
   - 原版分裂将三轴同除以 `1.6` 无法改变长宽比；改进后对针状高斯**仅将最长轴除以 2**，切分一次长宽比立即减半，并在 `densify_and_prune` 中支持 `--prune_needle_ratio` 与 `--max_ws_scale_ratio`，同时修复 `load_ply` 漏初始化 `max_radii2D` 的 Bug。
3. **做法 3（后处理几何精滤 + 固定拓扑表面愈合微调）** —— [`tools/clean_gaussians.py`](../tools/clean_gaussians.py)
   - 剔除残余超标长条与低透明雾点，冻结拓扑微调 1,000 步并硬性截断 `s_max <= ratio * s_mid`，无缝填平删点空隙。

### 实测效果（笔记本 `gc_data` 11 视角全分辨率）

| 指标 | 原始 DDGS | **DDGS + 做法 1 + 2 + 3 精修后** |
| :--- | :---: | :---: |
| 总高斯数 | 235,074 | **222,751** |
| 针状毛刺 (`s_max / s_mid > 10`) | 26,178 根 | **0 根** |
| 极端长刺 (`s_max / s_mid > 20`) | 9,499 根 | **0 根** |
| 全场最大针状比 | 1,495,579.88 | **8.00（100% 达标）** |
| **PSNR ($\uparrow$)** | 35.08 dB | **36.36 dB (+1.28 dB)** |
| **LPIPS ($\downarrow$)** | 0.1736 | **0.1678** |

---

## 七、探索实验：低清文字细节恢复（超分 vs AI OCR 矢量重绘边界分析）

针对经压缩后单字仅 ~6px 的实拍图像，对比测试了两条 2D 增强路线：
1. **通用超分（`Real-ESRGAN x4plus`）**：
   - **优势**：对 20~30px 的**物理键盘键帽大字母**修复极佳，与塑料材质颗粒感自然融合零违和。
   - **局限**：对 6px 屏幕密集代码与小汉字会出现**笔画粘连“鬼画符”或幻觉改字**（如把 `loss` 脑补成 `loos`），且多视角独立超分易引入跨视角不一致。
2. **透视矫正 + AI OCR 语义纠错 + 4K 矢量重绘回投**：
   - **优势**：可将 6px 模糊小字恢复至 100% 可读、语法高亮精准的矢量锐度。
   - **局限**：擦除旧字时的局部平滑会抹掉相机 CMOS 传感器底噪（ISO Grain）与屏幕自发光自然辉光（Halation），放大观察会有局部“补丁粘贴感”。
3. **工程最佳实践结论**：
   - 物理结构与大号字符可用 Anchor-View 轻量超分辅助；极小平面文字最自然无痕的方案是在采集端**用原相机（禁压缩）补充 1~2 张近景微距特写**送入统一 COLMAP + 3DGS 多尺度重建。

---

## 八、核心改进二：MW2 最优传输原理解析 & 自定义 CUDA `meanDistCUDA6` 算子加速

### 问题定位
[`scene/gaussian_model.py`](../scene/gaussian_model.py) 中的 `update_density_score()` 每 500 步将 `_xyz` 拷回 CPU 跑单线程 `sklearn.neighbors.NearestNeighbors(n_neighbors=6)` 再拷回 GPU，在大场景下每次停顿数秒。

### CUDA 算子实现
在 [`submodules/simple-knn/`](../submodules/simple-knn) 中复用 3D 莫顿码（Morton Z-order）基数排序与 1024 点 AABB 包围盒裁剪架构，新增模板核函数 `boxMeanL2DistK<6>` 与 PyBind11 接口 `meanDistCUDA6`（并修复原版边界框归约初值问题），使 `update_density_score` 变为一行纯 GPU 调用：
```python
self.density_score = -meanDistCUDA6(self.get_xyz.detach())
```

### RTX PRO 3000 Blackwell 实测基准对比

| 点云规模 | 原版 CPU `sklearn`<br>*(KD-Tree)* | GPU `PyKeOps`<br>*($O(N^2)$ 暴力归约)* | **自定义 CUDA `meanDistCUDA6`**<br>*($O(N \log N)$ 莫顿码包围盒)* | 加速比 & 精度一致性 |
| :--- | :---: | :---: | :---: | :--- |
| **63,155 点** (`trex`) | 158.18 ms | 9.81 ms | **14.01 ms** | **11.3×**（相关系数 `1.00000000`） |
| **235,074 点** (`gc_data`) | 594.48 ms | 106.74 ms | **32.69 ms** | **18.2×**（比 PyKeOps 快 3.3×） |
| **600,000 点** (MipNeRF360 级) | 3,136.74 ms | 721.19 ms | **113.34 ms** | **27.7×**（比 PyKeOps 快 6.4×） |
