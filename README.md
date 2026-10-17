# ScholarAlign

在单张 RTX 4090（24GB）上，用 QLoRA 对 Qwen2.5-7B-Instruct 做科研摘要和论文问答后训练。数据是 SciTLDR-AIC 与 QASPER 的官方划分。顺序是 SFT、DPO，再在 QASPER 上做 GRPO。

指标、同一测试集对比、为什么选 GRPO，以及为什么把这份 adapter 合并成 BF16 模型，写在 [数据分析.md](数据分析.md)。原始数字在 `数据记录.md`。

## 安装环境

- Python 3.10
- GPU：NVIDIA RTX 4090 24GB
- 命令都在项目根目录执行

```bash
pip install torch transformers datasets accelerate peft bitsandbytes trl rouge-score bert-score
```

基座放到 `models/base/Qwen2.5-7B-Instruct`，目录内要有 `config.json` 和分词器。训练脚本只读本地文件。

## 目录

```text
data/raw/             官方原始文件
data/interim/         清洗后的 SciTLDR、QASPER
data/processed/       截断后的 SFT jsonl 和 DPO 偏好对
data/stats/           token 统计和 frozen_config.json
models/base/          Qwen2.5-7B-Instruct
models/sft_adapter/   SFT LoRA
models/dpo_adapter/   DPO LoRA
models/grpo_adapter/  GRPO LoRA
outputs/              各阶段 metrics、predictions、checkpoint
scripts/              下面列出的核心脚本
```

`models/` 和 `outputs/` 里的权重不要提交到 GitHub。

## 训练时不要改的配置

这些值写进 `data/stats/frozen_config.json` 之后保持不动。

- cutoff 4096。超长样本截断，不删除。
- 评测用官方 test。生成长度 512，贪心解码。
- QLoRA：NF4，BF16 计算，r=16，alpha=32，dropout=0.05。目标模块是 q/k/v/o 和 gate/up/down。
- 批大小 1，梯度累积 16，优化器 `paged_adamw_8bit`，种子 42。
- SFT：学习率 2e-4，2 个 epoch，SciTLDR 和 QASPER 一起训。
- DPO：从 SFT adapter 继续，学习率 5e-6，1 个 epoch，beta 0.1，`max_length=4096`，`truncation_mode=keep_end`。
- GRPO：只训 QASPER，学习率 1e-6，每条 prompt 采样 4 次，`steps_per_generation=4`。奖励公式在 `scripts/rewards.py`，不要改权重。

SciTLDR 一条论文一个样本，答案用 `references[0]`。QASPER 一个问题一个样本。DPO 的 rejected 用规则破坏，不用测试集上的模型输出。

## 训练步骤

### 1. 下载和清洗

```bash
python scripts/download_data.py
python scripts/build_qasper.py
python scripts/build_scitldr.py --dataset data/raw/scitldr/hf_dataset
```

`download_data.py` 得到的 SciTLDR 是 jsonl。`build_scitldr.py` 读的是 Hugging Face `DatasetDict`，默认目录 `data/raw/scitldr/hf_dataset`，需要 train、validation、test。已有 `data/interim/scitldr_clean.jsonl` 时不用重跑清洗。

### 2. 冻结长度并打包

```bash
python scripts/analyze_scitldr.py
python scripts/analyze_qasper.py
python scripts/build_sft_dataset.py
```

先跑 SciTLDR 的 token 统计。`analyze_qasper.py` 才会写出 `frozen_config.json`。

### 3. 基座评测和 SFT

```bash
python scripts/evaluate.py
python scripts/train_sft.py
python scripts/evaluate.py --adapter models/sft_adapter --output outputs/sft --load-in-4bit --overwrite
```

### 4. 偏好数据和 DPO

```bash
python scripts/build_preference.py
python scripts/train_dpo.py
python scripts/evaluate.py --adapter models/dpo_adapter --output outputs/dpo --load-in-4bit --overwrite
```

### 5. GRPO

```bash
python scripts/rewards.py
python scripts/train_grpo.py
python scripts/evaluate.py --adapter models/grpo_adapter --output outputs/grpo --load-in-4bit --overwrite
```

`rewards.py` 不通过就不要开始 GRPO。训练中断后，只有 `outputs/grpo/checkpoint-*` 存在时才能续上：

```bash
python scripts/train_grpo.py --resume
```

续训回到最近一次完整存盘，两次存盘之间的步数不会补回。

## 核心训练文件

| 文件 | 做什么 | 干什么 |
|---|---|---|
| `scripts/build_sft_dataset.py` | 把清洗后的 SciTLDR 和 QASPER 打成 SFT jsonl | 按冻结的 4096 截断，写出 train/val/test，后面所有训练都读这批文件 |
| `scripts/evaluate.py` | 在冻结测试集上贪心生成并打分 | 基座、SFT、DPO、GRPO 用同一套 ROUGE、BERTScore、Answer F1、Evidence F1、Unsupported |
| `scripts/train_sft.py` | QLoRA-SFT | 让模型先学会摘要和 JSON 问答格式，得到 `models/sft_adapter` |
| `scripts/build_preference.py` | 构造 DPO 偏好对 | chosen 用训练/验证金标，rejected 用规则破坏，测试集不进入 |
| `scripts/train_dpo.py` | 从 SFT adapter 做 DPO | 提高金标、压低规则负例，得到 `models/dpo_adapter` |
| `scripts/rewards.py` | QASPER 的可验证奖励 | 用答案 F1、证据 F1、格式、无依据和长度打分；单元测试不过就停止 |
| `scripts/train_grpo.py` | 从 DPO adapter 做 GRPO | 只在 QASPER 上采样 4 条并按组内奖励更新，得到 `models/grpo_adapter` |
