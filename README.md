# GeoRadioTwin

Geometry-guided continuous radio-field prediction for sampling-free radio-map experiments.

本仓库整理了 GeoRadioTwin 的模型、训练、统一评估、基线、消融和数字孪生能力验证代码。当前研究版本面向 ICC 2027；论文尚未声明接收或发表。主方法在推理时只使用建筑高度、发射机状态与接收机查询位置，不需要实测无线电采样点。

## 输入、输出与流程

输入为建筑高度栅格、Tx 坐标/高度、Rx 坐标/高度。代码生成 LoS 与首遮挡深度特征，用连续坐标网络预测基础场，再通过建筑 CNN 和射线路径上下文修正预测。输出为连续查询位置的无线电图编码值，可重建完整地图。

```text
UrbanRadio3D 文件 + 建筑高度图
    -> 原始数据清单 -> 场景互斥划分 -> 固定均衡地图子集
    -> LoS/Depth 缓存 -> 连续场预训练 -> HybridGeo 训练
    -> 同一测试掩码评估 -> 三种子汇总
    -> Tx/高度状态响应、规划、h5-h8 泛化、集成不确定性
```

训练高度为 h1–h4，未见高度评估为 h5–h8。数字孪生验证覆盖状态条件查询与场景内规划；建筑编辑实验需独立仿真标签，默认运行流程不包含这类实验。

## 目录

| 目录 | 用途 |
|---|---|
| `configs/` | 主方法、基线与消融配置 |
| `datasets/` | 原始文件解析、坐标查询和稠密基线数据加载 |
| `geometry/` | 射线几何查询与缓存 |
| `models/` | 连续场、HybridGeo 与基线适配器 |
| `physics/`, `calibration/` | 物理正则与历史校准模块 |
| `scripts/` | 训练、评估、复现入口与论文图表生成 |
| `utils/` | 配置、指标、随机种子、检查点等 |
| `tests/` | 使用合成/临时数据的单元测试 |
| `docs/` | 实验协议、基线适配说明与审计记录 |
| `external/` | 上游来源登记，不包含第三方代码 |
| `results/` | 已完成实验的冻结汇总 CSV |
| `paper_artifacts/` | 数值源表与 LaTeX 表格 |

## 环境与数据

推荐 Python 3.10+。正式训练使用支持 CUDA 的 PyTorch；请先按 [PyTorch 官方说明](https://pytorch.org/get-started/locally/)安装与你的 GPU/驱动匹配的版本，然后安装其余依赖。

```powershell
conda create -n georadiotwin python=3.10 -y
conda activate georadiotwin
# 先安装匹配本机 CUDA 的 PyTorch
python -m pip install -r requirements.txt
python -c "import sys,torch; print(sys.executable); print(torch.__version__); print('CUDA:',torch.cuda.is_available())"
```

原始数据需从 [UrbanRadio3D 上游项目](https://github.com/UNIC-Lab/UrbanRadio3D)单独获取，并遵守数据提供方条款。本仓库不分发数据、模型权重或几何缓存。

每次运行前配置实际数据根目录：

```powershell
$env:GEORADIOTWIN_PATHLOSS_ROOT = 'D:\data\UrbanRadio3D\PathLoss'
$env:GEORADIOTWIN_BUILDING_ROOT = 'D:\data\UrbanRadio3D\buildingsWHeight'
```

期望目录示例：`PathLoss_h1_train/h1/0_X101_Y75.png` 和建筑目录中的 `0.png`。环境变量是必填项；公开版不依赖作者电脑的盘符。坐标约定为 `row = H - 1 - X`、`col = Y`，X 向上、Y 向右。

## 先做本地检查

```powershell
python -m unittest discover -s tests -v
python scripts/prepare_raw_dataset.py --config configs/base.yaml --inspect-only --inspect-count 8
python scripts/prepare_raw_dataset.py --config configs/base.yaml
```

数据清单与场景划分写入 `data/processed/`，不会修改原始 PNG。没有外部模型源码时，相关可选测试会标记跳过。

## 训练主方法并评估

```powershell
python scripts/build_geometry_cache.py --config configs/base.yaml --split train --max-maps 4096 --selection-strategy balanced_scene
python scripts/build_geometry_cache.py --config configs/base.yaml --split val --max-maps 128 --selection-strategy balanced_scene
python scripts/train.py --config configs/coord_los_depth.yaml
python scripts/train_hybrid.py --config configs/hybrid_geo.yaml
python scripts/evaluate_hybrid.py --config configs/hybrid_geo.yaml
```

HybridGeo 使用 LOS+Depth 阶段最佳检查点初始化。训练完成后以最佳验证 NMSE 检查点评估测试集，不以测试指标选择模型。中间文件写入 `outputs/` 与 `cache/`。

## 三种子、基线和消融

UniRM 基线单独获取固定版本上游源码，不能将其 checkout 提交到本仓库：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_external_baselines.ps1
powershell -ExecutionPolicy Bypass -File scripts\run_icc_repro_suite.ps1 -PythonExecutable (Get-Command python).Source
```

该流程保持数据选择种子 42，训练种子为 42、43、44，包含主方法、UniRM、PEFNet-inspired、RadioUNet-style、RMTransformer-style、RadioDiff-3D-style、拟合 log-distance 及结构消融。已完成任务默认复用。正式完整运行可能需要大量 GPU 时间；`-SmokeTest` 可验证短流程。PMNet 和稀疏采样相关模块作为历史代码保留，不在当前主结果比较中。

第三方比较结果来自共同协议下的适配，不能描述为原论文性能或官方完整复现。具体来源和假设见 `external/BASELINE_SOURCES.json`、`docs/PUBLIC_BASELINES.md`、`docs/PEFNET_ADAPTATION.md`。

## 孪生能力验证与一键流程

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_twin_validation.ps1 -PythonExe (Get-Command python).Source
powershell -ExecutionPolicy Bypass -File scripts\run_twin_completion_experiments.ps1 -PythonExecutable (Get-Command python).Source
```

准备好数据清单和 UniRM 源码后，训练、汇总、验证、图表与单元测试可用一个入口执行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_full_paper_pipeline.ps1 -PythonExecutable (Get-Command python).Source
```

## 统一指标与已完成结果

训练目标为原始像素值除以 255；论文主指标将预测与标签统一转为 `v / 127.5 - 1`，在有效 Rx 像素上逐图计算，然后对固定 128 张测试地图宏平均。三种子结果报告均值与样本标准差。RMSE/MAE 为归一化编码误差，NMSE 为线性比值；只有 PSNR 用 dB。原始 PNG 到物理路径损耗 dB 的映射未验证，不能把上述误差称为物理 dB 误差。

主方法有 1,093,154 个参数；冻结汇总的三种子结果为 NMSE `0.02574 ± 0.00057`、RMSE `0.07422 ± 0.00065`、PSNR `29.23 ± 0.08`。完整基线和消融见 `results/aggregate.csv`，逐次运行见 `results/per_run.csv`。这些是已有实验快照，不代表下载代码后已在你的机器上重跑。

## 发布范围与许可

本版本包含研究代码和汇总数值，不包含原始场景图、权重、缓存、登录信息、论文草稿或第三方 checkout。尚未为原创代码选择开源许可证，见 `LICENSE-NOTICE.md`。上游资源的条款不因本仓库上传而改变。

