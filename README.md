# ComplexMD：蛋白质–小分子复合物轨迹预测

ComplexMD 是 GOAI 2026「小分子–蛋白质结合轨迹预测」任务的可复现实现。模型不预测原子受力，也不数值积分牛顿方程；它在由观测帧定义的蛋白质口袋参考系中，联合预测：

1. 蛋白质复合物的全局 SE(3) 平移与旋转；
2. 小分子相对口袋的条件 Rectified Flow 运动；
3. 符合原始 PDB 原子顺序的全原子未来 XTC。

Python 包仍名为 `bindmd`，以保持已有 checkpoint 的兼容性。

## 1. 仓库结构

```text
ComplexMD/
├── README.md
├── run.sh                         # 无参数一键推理入口
├── environment.yml               # 推荐且唯一的主环境安装方式
├── checkpoints/epoch_015.pt      # 最终推理权重
├── configs/complexmd_inference.yaml
├── bindmd/                        # 数据、模型和指标实现
├── scripts/predict_goai.py        # 全原子 XTC 推理
├── scripts/validate_goai_submission.py
├── scripts/package_submission.py
├── tests/
└── docs/
```

本仓库不包含 MISATO、GOAI 评测数据、训练缓存或预测答案。

## 2. 环境安装

唯一推荐安装方式为 Conda。已验证环境为 Linux、Python 3.9.25、PyTorch 2.2.0、CUDA 12.1、PyG 2.5.2 和 MDAnalysis 2.7.0。

```bash
conda env create -f environment.yml
conda activate complexmd
pip install -e . --no-deps
```

最低建议硬件：1 张支持 CUDA 的 GPU、16 GB 显存、4 核 CPU、32 GB 内存和 10 GB 可用磁盘。最终实验使用 NVIDIA A100 40 GB。CPU 可以加载代码，但完整 T3 推理不建议使用 CPU。

## 3. 评测数据准备

评测数据由赛事组委会提供，任务定义以评测包中的 `README.md` 和 `protocol.json` 为准。将公开或独立核验数据放到仓库根目录：

```text
GOAI_eval_public/
├── protocol.json
├── T1/ids.txt
├── T1/T1-1/T1-1.pdb
├── T1/T1-1/T1-1_obs.xtc
├── T1/T1-1/meta.json
├── T2/...
└── T3/...
```

`run.sh` 会依次自动查找：

1. 环境变量 `GOAI_INPUT_ROOT` 指向的位置；
2. 仓库根目录下的 `GOAI_eval_public/`；
3. `/data/GOAI_eval_public/`。

正常复现时无需修改任何源码或脚本参数。

## 4. 权重与关键配置

- 权重：`checkpoints/epoch_015.pt`
- SHA256：`96fff72a87d7c9a7b24f59501a317f8443b2a1fb612b7c6e2602a1739f871616`
- 配置：`configs/complexmd_inference.yaml`
- 随机种子：42；对第 `i` 个体系使用 `42 + i`
- Flow 求解器：Heun，10 个采样步
- 蛋白质位姿平移残差缩放：0.25
- 蛋白质位姿旋转残差缩放：0.25
- 小分子内部形变缩放：0，即当前版本保持内部刚性

权重约 30 MB。`.gitattributes` 已将 `*.pt` 配置为 Git LFS 文件；若 GitHub 仓库不使用 LFS，也可将权重上传为固定 Release 资产，但必须保持上述文件名和 SHA256，或同步修改 README 与 `run.sh` 中的稳定下载逻辑。

## 5. 一键推理

组委会只需从仓库根目录执行：

```bash
bash run.sh
```

脚本全程非交互，依次完成：

```text
环境/权重/数据自检
→ 读取观测轨迹
→ 口袋对齐和条件编码
→ T1、T2、T3 递推预测
→ 恢复完整蛋白质与小分子坐标
→ 写出 nm 单位 XTC
→ 校验帧数、原子数和有限坐标
→ 生成材料 A 压缩包
```

可选的 `GOAI_INPUT_ROOT`、`COMPLEXMD_OUTPUT_ROOT` 和 `COMPLEXMD_PYTHON` 环境变量只用于不同机器的路径适配；默认目录结构下无需设置。

## 6. 运行产物

成功后生成：

```text
predictions/
├── T1/T1-1_pred.xtc ... T1-30_pred.xtc
├── T2/T2-1_pred.xtc ... T2-30_pred.xtc
├── T3/T3-1_pred.xtc ... T3-30_pred.xtc
└── validation.json

GOAI_pred_COMPLEXMD.zip
```

压缩包第一层直接是 `T1/`、`T2/`、`T3/`，只包含 90 条 `*_pred.xtc`，不含额外父目录、观测帧或缓存文件。T1、T2、T3 的未来帧数分别为 10、20、80，原子数和顺序与对应 PDB 完全一致，XTC 单位为 nm。正式提交时请按队伍 ID 将压缩包改名为 `GOAI_pred_<队伍ID>.zip`。

在 A100 40 GB 上的实测参考耗时为：T1 约 2 分钟、T2 约 3–5 分钟、T3 约 14–18 分钟，单 GPU 完整运行约 20–25 分钟。不同体系原子数和磁盘性能会影响耗时。

## 7. 方法与推理流程

每个观测体系首先使用蛋白质口袋骨架对所有帧做 Kabsch 对齐，并以第一帧定义唯一参考方向。模型由口袋编码器、联合时空注意力模块、小分子条件 Flow 头和蛋白质位姿头组成。

小分子头直接学习从简单基分布到下一帧真实相对位移的连续速度场；蛋白质头预测相邻帧的平移和轴角旋转增量。推理时递推生成未来帧，再将预测的全局 SE(3) 变换作用于完整蛋白质和小分子。小分子重原子确定一个刚体变换，氢原子跟随同一变换。整个流程只读取 PDB、`meta.json` 和 `*_obs.xtc`，不会读取或检索未来轨迹。

## 8. 训练数据与外部资源披露

- 训练数据：MISATO 的 NeuralMD 半柔性预处理表示；13,066 个训练体系、1,357 个验证体系、1,357 个测试体系。
- 划分：沿用 NeuralMD/MISATO 固定划分；GOAI 评测体系及其未来帧未参与训练或调参。
- 基础模型：无预训练基础模型。
- 外部势函数：无。
- 外部 MD 数据：除上述 MISATO 训练轨迹外无其他 MD 数据。
- 方法参考：NeuralMD 的数据契约与评估指标、STAR-MD 的联合时空建模思想、ConfRover 的自回归条件结构生成思想。本仓库实现为独立代码，并未复制未开源的 STAR-MD 仓库。
- 许可证：本代码使用 MIT License。MISATO、NeuralMD、ConfRover 和赛事评测数据不在本仓库再分发，使用者应分别遵循其官方来源和许可证/使用条款。

## 9. 自检与测试

`run.sh` 在推理前检查 Python、CUDA、数据目录、T1–T3 的 `ids.txt`、配置文件以及 checkpoint SHA256；推理后逐条读回 90 个 XTC，检查帧数、原子数和 NaN/Inf，并只在全部通过后打包。

独立单元测试运行方式：

```bash
python -m pytest -q
```

当前测试覆盖坐标对齐、SE(3) 几何、Flow、分层位姿头、指标和赛事 XTC 输出契约。

## 10. 已知限制与常见问题

- 当前小分子内部形变被设为零，因此能稳定保持键长，但不能显式生成真实扭转角变化。
- 蛋白质内部结构使用第一帧模板，只预测复合物全局平移和旋转；不预测口袋内部柔性。
- 长时程 T3 会积累旋转误差，位姿残差缩放 0.25 是验证集选择的保守设置。
- 公开评测包没有未来真值，因此本仓库只能做观测段因果回测，不能提前给出官方隐藏测试分数。
- 若报 checkpoint 哈希错误，请确认 Git LFS 已执行 `git lfs pull`，而不是只下载到了 LFS 指针文件。
- 若 MDAnalysis 读取 XTC 报原子数不一致，请确认 PDB、观测 XTC 与 `meta.json` 来自同一个评测体系且未改动原子顺序。

完整阶段结果见 `docs/RESULTS.md`，架构细节见 `docs/DESIGN.md`。

## 11. 结果摘要

在 1,357 个 MISATO 测试复合物上，ComplexMD 在 T1/T2/T3 协议下的 RMSE 分别为 1.2244、1.4277 和 2.0323；旧 BindMD 为 1.2337、1.4468 和 2.1150；NeuralMD 为 2.1738、2.7992 和 4.8086。比较边界和完整指标见 `docs/RESULTS.md`。

