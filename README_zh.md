# HERNet 使用说明

本目录提供论文 **Hierarchical Spatial Routing of Heterogeneous Experts for Latent Fingerprint Enhancement** 的网络定义、推理权重、文件夹批量增强入口、训练数据准备工具和补充识别曲线。

## 快速使用

安装适合当前 CPU/CUDA 环境的 PyTorch，再安装依赖：

```bash
python -m pip install -r requirements.txt
python enhance.py --input /path/to/latent_images --output /path/to/enhanced_images --device cuda:0
```

没有 GPU 时使用 `--device cpu`。默认权重位于 `weights/hernet_inference.pth`，可通过 `--weights` 指定。`--recursive` 支持子目录；输出为原尺寸灰度 PNG。输入和输出目录应分开，已有输出默认不会覆盖。

输入应为8位图像，保持与实验一致的脊线极性和约500 ppi尺度。模型不会自动反色、分割或重采样。RGB输入按通道平均转换为灰度。推理保留原有填充、逐图标准化和裁剪方式，权重中不包含训练专用辅助预测头。

## 文档和结果

- [完整使用说明](README.md)
- [训练图像生成及HDF5转换](docs/training_data.md)
- [四种匹配流程及评价口径](docs/matching_protocol.md)
- [六种增强基线的实施说明](docs/baselines.md)
- [CMC总览](supplementary/cmc/CMC_all_datasets_matchers.png)及[数值文件说明](docs/cmc_results.md)

基准数据、第三方模型及商业匹配器不随本目录分发。生成图像的视觉清晰度不等同于真实结构恢复，增强结果不能代替对原始指纹的法证检验。
