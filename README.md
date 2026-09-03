# ComplexMD - GOAI 2026 复赛可复现版本

本仓库是 GOAI 2026 赛道三“小分子-蛋白质结合轨迹预测”任务的冻结可复现版本。项目已由 BindMD 更名为 ComplexMD；为保持已有模型权重可直接加载，内部 Python 包名仍保留为 `bindmd`。

评审人员在仓库根目录只需执行一条非交互命令：

```bash
bash run.sh
```

## 1. 任务定义与预期输出

输入仅包含蛋白质-小分子复合物的已观测全原子轨迹、PDB 拓扑和 `meta.json`。ComplexMD 只预测未来帧，不读取或输出观测段。程序运行后生成：

```text
predictions/
  T1/T1-1_pred.xtc ... T1-30_pred.xtc
  T2/T2-1_pred.xtc ... T2-30_pred.xtc
  T3/T3-1_pred.xtc ... T3-30_pred.xtc
run_logs/
  manifests/T1/prediction_manifest.json
  manifests/T2/prediction_manifest.json
  manifests/T3/prediction_manifest.json
  submission_validation.json
```

每条 XTC 轨迹满足以下要求：

- 只包含未来的 `meta.json:n_pred` 帧；
- 原子数量和原子顺序与对应 PDB 完全一致；
- 坐标单位为 nm；
- T1-T3 共生成 90 条 `*_pred.xtc` 轨迹。

本冻结版本不生成可选加分项 T4。

## 2. 评测数据放置方式

推荐将组委会评测数据解压到仓库内的以下默认位置：

```text
ComplexMD_GitHub_20260902/
  evaluation_data/
    protocol.json
    T1/
    T2/
    T3/
```

`run.sh` 还会自动识别以下目录：

```text
./GOAI_eval_public
./public
./data/evaluation_data
../GOAI_eval_public
```

如果评测平台必须将数据挂载到其他路径，可以预先设置环境变量 `GOAI_INPUT_ROOT`，随后仍执行同一条无参数命令 `bash run.sh`，不需要修改 Python 源码。

本仓库不重新分发任何评测数据。

## 3. 环境安装

主要且经过测试的环境安装方式为：

```bash
conda env create -f environment.yml
conda activate complexmd-goai
```

测试环境如下：

| 依赖 | 版本 |
|---|---:|
| Python | 3.9.25 |
| PyTorch | 2.2.0 |
| CUDA 运行时 | 12.1 兼容版本 |
| PyTorch Geometric | 2.5.2 |
| NumPy | 1.26.4 |
| MDAnalysis | 2.7.0 |
| h5py | 3.14.0 |
| PyYAML | 6.0.2 |

XTC 的读取和写出由 MDAnalysis 完成，不需要额外安装 GROMACS。

## 4. 最终模型与配置

- 模型权重：`weights/complexmd_v3_6plus6_epoch004.pt`
- 权重 SHA256：`9493faa931d305ec3a78b4c14a1e6a3257d400fc9114a935bdab9606c81901ee`
- 推理配置：`configs/complexmd_v3_6plus6_submission.yaml`
- 随机种子：42，并根据体系顺序确定性偏移
- 输入窗口：12 帧，其中前 6 帧编码较早历史趋势，后 6 帧编码近期局部动力学；观测帧少于 12 帧时只进行因果填充
- 生成方法：条件 Flow Matching
- 数值求解器：Heun，10 个采样步
- 配体拓扑：默认使用评测 PDB 中的 `CONECT` 键连关系
- `QM.hdf5` 仅用于训练阶段提供键级和杂化信息，不是比赛推理依赖；当评测输入没有杂化标签时，程序使用保留的 unknown 编码并根据 `CONECT` 保守构建刚体片
- 扭转置信度门控：置信度低于 0.75 的扭转角自动收缩为 0
- 扭转角限制：单步最大 5°
- 蛋白质位姿平移/旋转缩放：0.25/0.25

代码支持可选的本地 SMILES 映射，但最终 `run.sh` 不调用 SMILES 数据库，也不进行任何在线检索。

训练目录中的 `epoch_004.pt` 与 `last.pt` 模型参数逐张量一致。本版本选择已被完整评估记录明确引用的 `epoch_004.pt`，便于版本审计。

## 5. 方法与坐标处理流程

对于每个复合物，程序首先使用 Kabsch 算法将所有已观测帧的蛋白质口袋骨架对齐到第 0 帧。对齐后，模型在统一的参考坐标系内联合编码裁剪后的蛋白质口袋、配体原子特征和时间特征。

模型包含三个相互耦合的预测部分：

1. 预测蛋白质口袋整体的 SE(3) 平移和旋转；
2. 通过条件 Flow Matching 预测小分子相对蛋白质口袋的整体 SE(3) 运动；
3. 预测自动识别的小分子刚体片之间的相对旋转。

刚体片根据输入中的键连关系自动识别。对于片段间的可旋转键，子片段及其下游原子作为整体，使用精确的 Rodrigues 公式绕键轴旋转，因此刚体片内部的原子间距离不会因旋转而漂移。置信度门控用于抑制不确定的扭转运动。

生成对齐坐标后，模型从最后一个观测位姿开始累积蛋白质位姿增量，并使用同一世界坐标变换恢复蛋白质和相对配体坐标。配体氢原子跟随对应的刚体片或整体刚体变换；未单独建模的蛋白质原子保持第 0 帧内部模板。坐标恢复、单位转换和 XTC 写出均已包含在 `run.sh` 中，不需要人工执行坐标转换、文件重命名或轨迹拼接。

## 6. 一键推理与自动检查

执行：

```bash
bash run.sh
```

该命令完成以下完整流程：

```text
环境与文件检查
  -> 权重和配置校验
  -> PDB/XTC 读取
  -> 第0帧口袋对齐
  -> 键连与刚体片构建
  -> 自回归轨迹生成
  -> 世界坐标恢复
  -> XTC写出
  -> 90条轨迹格式校验
```

推理前自检包括：

- 检查模型权重 SHA256；
- 严格加载模型参数；
- 检查 6+6 窗口和最终推理阈值；
- 检查 T1-T3 共 90 个输入体系；
- 检查每个体系的 `n_obs`、`n_pred`、PDB 和观测 XTC。

推理完成后，验证器会重新读取每条预测 XTC，并检查文件名、帧数、原子数以及坐标是否包含 NaN 或无穷值。验证成功时：

```text
run_logs/submission_validation.json
```

中的 `valid` 字段应为 `true`。

可选单元测试：

```bash
python -m pytest -q
```

当前服务器版本共 32 项测试，均已通过。

## 7. 硬件条件与运行时间

推荐配置：

- NVIDIA A100 或同等级 GPU；
- 至少 24 GB 可用显存；
- 8 核 CPU；
- 32 GB 内存。

模型权重约为 37.4 MiB。模型采用逐帧自回归生成，T3 的 80 帧长期预测占据主要推理时间。具体时间取决于体系原子数量和 GPU 状态，建议为全部 90 个体系预留数小时。代码可以回退到 CPU，但不建议使用 CPU 完成全量评测。

## 8. 训练数据、数据划分与外部资源披露

- 训练数据基座：MISATO（Nature Computational Science，2024）。
- 数据划分：采用 MISATO 原始 MD 划分，并剔除 NeuralMD `peptides.txt` 中的多肽体系。
- 数据规模：训练集 13,066 个复合物，验证集和测试集各 1,357 个复合物。
- 权重初始化：6+6 版本从前一版 ComplexMD v2 权重继续微调。
- 训练和推理均未使用组委会隐藏轨迹，也未读取评测体系的未来帧。
- 外部方法参考：NeuralMD 的数据表示与指标归约、STAR-MD 的时空联合建模思想，以及 ConfRover 风格的自回归结构生成。
- ComplexMD 为独立实现，不重新分发上述项目的权重。
- 推理过程中不使用商业 API、外部势函数、在线 MD 数据检索或基于体系标识的答案检索。
- 最终入口仅使用评测输入 PDB 自带的 `CONECT` 信息构建配体拓扑，不调用外部 SMILES 数据库。

MISATO 和参考项目继续遵循各自原始许可证与引用要求。ComplexMD 本仓库代码采用 MIT 许可证，详见 `LICENSE`。

## 9. 实验结果与指标口径

`results/README.md` 汇总了 ComplexMD 6+6 与 NeuralMD 在完整 MISATO 测试集上的 T1/T2 对比，以及各指标的定义和限制。逐体系机器可读结果位于：

```text
results/final_6plus6/T1.json
results/final_6plus6/T2.json
```

这些结果文件是实验佐证，不参与推理，也不是比赛材料 A 所需的预测 XTC。当前 T3 分片评估尚未完整结束，因此本冻结版本不声明 T3 全量聚合指标。

由于组委会没有公开隐藏测试集的归一化常数，本仓库只报告原始指标，不自行构造非官方综合分数。

## 10. 已知限制与常见问题

- 模型预测蛋白质口袋整体位姿，但蛋白质内部形变保持为第 0 帧模板。
- 刚体片识别质量依赖输入 PDB 的键连信息；最终推理入口不要求额外提供 SMILES。
- T4 使用 1 ns 帧间隔，与本版本训练采用的 80 ps 分辨率不同，因此不在当前冻结提交范围内。
- 出现 `checkpoint SHA256 mismatch`：权重文件不完整或版本错误，请替换为第 4 节指定的权重。
- 出现 `evaluation data were not found`：将数据放入 `evaluation_data/`，或设置 `GOAI_INPUT_ROOT`，无需修改源码。
- CUDA 显存不足：关闭其他 GPU 任务。提交入口逐体系推理，不需要修改 batch size。
- 重复运行会覆盖同名预测 XTC；诊断 JSON 始终写在材料 A 目录之外的 `run_logs/`。

## 11. 仓库目录结构

```text
bindmd/          模型、几何运算、拓扑处理和GOAI数据适配器
configs/         原始训练配置和冻结推理配置
scripts/         训练、评估、推理、预检和输出校验脚本
tests/           几何、指标、输出格式和等变性测试
weights/         最终6+6模型权重
results/         完整测试集实验记录
docs/            设计说明和消融实验记录
run.sh           无参数一键推理入口
environment.yml  可复现环境定义
```

