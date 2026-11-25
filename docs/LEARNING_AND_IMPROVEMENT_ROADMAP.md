# 未来学习路线与算法/工程改进计划（Roadmap）

基于前期对 **D²GS** 架构的拆解与两项已落地的改进（**针状毛刺根治管线** + **`meanDistCUDA6` GPU 算子加速**），制定以下四个阶段的系统化深入学习与二次创新路线：

---

## 阶段一：下探 `diff-gaussian-rasterization` 光栅化 CUDA 源码（底层精读）

- [ ] **前向渲染核函数精读**（[`submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu`](../submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu)）
  - 推导 EWA Splatting 雅可比矩阵 $J$ 将 3D 协方差 $\Sigma$ 投影为 2D 屏幕协方差 $\Sigma'$ 的精确公式（`computeCov2D`）；
  - 理解 `16×16` Tile 划分、64 位键值（高 32 位 Tile ID + 低 32 位深度）基数排序（`DeviceRadixSort`）与 Shared Memory 协同 Alpha 混合提前终止机制（透射率 $T < 10^{-4}$ 退出）。
- [ ] **解析反向梯度核函数精读**（[`submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu`](../submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu)）
  - 理解为何反向传播从后往前遍历高斯序列恢复透射率 $T_i$，以及屏幕空间梯度 `dL_dmeans2D` 如何触发 `GaussianModel` 的分裂与克隆判据。
- [ ] **动手改造练习：原生渲染深度图输出（摆脱外部 `preprocessed_masks` 依赖）**
  - 在 `forward.cu` Alpha 混合循环中同步累积期望深度 $D = \sum_i z_i \alpha_i T_i$ 返回给 Python 端；
  - 在 `train.py` 中直接利用模型自渲染深度动态生成 `far_mask`，使 DAFE 模块实现**零外部预处理、端到端自闭环训练**。

---

## 阶段二：微距高频纹理与多尺度抗锯齿改进（解决小字/细节模糊）

- [ ] **引入 Mip-Splatting 3D 平滑滤波 + 2D Mip 滤波器**
  - 替换 `forward.cu` 中硬编码的 `cov[0][0] += 0.3f` 低通滤波核，解决高低分辨率跨尺度照片（远景全局图 + 近景微距特写图）混训时的膨胀或走样伪影。
- [ ] **高频边缘感知损失与两阶段 DD-Drop 退火日程**
  - 在 `train.py` 增加可选的 Sobel/Laplacian 图像梯度损失 $\mathcal{L}_{\text{edge}}$，提升文字边缘和高频纹理权重；
  - 将 DD-Drop 的线性递增日程改进为 **“前 70% 步数升温防过拟合 $\rightarrow$ 后 30% 步数退火归零精雕细节”**，消除高密度文字区因持续随机丢弃导致的笔画变胖模糊。
- [ ] **自然无痕的场景文字增强探索（消除 2D 重绘粘贴感）**
  - 测试多尺度实拍融合（8~10 张全局照 + 2~3 张原相机无损微距特写联合 SfM + 3DGS）；
  - 探索基于扩散先验的场景文字修复（如 `AnyText` / `DiffTSR`）结合相机传感器噪声建模，消除纯矢量重绘的矩形块边界感。

---

## 阶段三：稀疏视角几何先验与相机位姿联合优化

- [ ] **引入单目深度几何正则（Depth Anything V2 + Pearson 相关损失）**
  - 对于白墙、纯色桌面等 COLMAP `patch_match_stereo` 易空洞的弱纹理区域，接入预训练单目深度图做尺度无关深度排序监督。
- [ ] **相机位姿联合微调（Pose Optimization / BARF 思想）**
  - 手持实拍照片在稀疏视角下 COLMAP 位姿往往存在 0.5~1.5px 重投影误差，导致 1px 宽的屏幕文字多视角错位重影；在训练后半程对相机外参 $(R, T)$ 开启微小学习率联合优化，对齐亚像素级多视角投影。

---

## 阶段四：模型压缩、网格提取与交互式部署应用

- [ ] **模型轻量化压缩**：对高阶球谐系数 `_features_rest` 做码本量化（Vector Quantization）与重要性剪枝，将 56 MB 的 `.ply` 压缩至 5~8 MB 便于网页秒开。
- [ ] **表面网格导出（2DGS / GOF 法向正则）**：增加深度-法向一致性约束，支持从训练好的高斯场一键 TSDF 融合导出干净的三角网格（`.obj` / `.glb`）。
- [ ] **实时交互查看器集成**：编译接入本地 `SIBR_viewers` 或 `nerfstudio/viser` Web 查看器，连接 `network_gui.py` (`127.0.0.1:6009`) 边训练边实时拖拽视角观察高斯演化。
