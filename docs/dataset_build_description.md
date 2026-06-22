# AdaRoute-MM 数据集构建说明

本文档说明当前 AdaRoute-MM 实验使用的两类数据集：文本融合数据集和图像问答数据集。文本数据集用于 text-only 路由与模型选择实验；图像数据集用于多模态 VQA、VLM caption cache、fallback 和端到端 AdaRoute-MM 实验。

## 1. 数据集总体设计

AdaRoute-MM 的实验数据按任务模态拆分为两条主线：

| 数据集 | 模态 | 主要用途 | 当前文件 |
| --- | --- | --- | --- |
| 文本融合数据集 | text-only | 验证文本路由、模型选择、成本/时延/准确率权衡 | `data/datasets/v3_1_text_fusion/fusion_v3_1_1000_200-200-400-100-100.jsonl` |
| VQAv2 图像问答数据集 | image + question | 验证 VLM caption、图像缓存、fallback 和多模态路由 | `data/datasets/vqav2_yesno_1000.jsonl` |

两类数据都统一转换为 JSONL 格式，每一行代表一个可独立推理与评估的样本。文本样本的 `image_path` 为空；图像样本包含本地图片路径，供 VLM 模块读取。

## 2. 文本融合数据集

### 2.1 构建目标

文本融合数据集用于 v3_1 及后续 text-only 实验，目标不是复现单一 benchmark 的完整排行榜，而是构造一个适合路由研究的 1000 条混合评估集。该数据集同时覆盖：

- 容易题：用于观察小模型可承担的样本；
- 中等难度科学问答：用于测试 medium 模型和路由边界；
- 挑战性选择题：用于检验 hard routing 是否能识别复杂样本；
- 阅读理解 span 抽取：用于测试长上下文和答案抽取；
- 数值/数学推理：用于保留大模型优势明显的样本。

### 2.2 数据来源与融合比例

当前文本融合数据集总规模为 1000 条，融合比例固定为：

```text
arc_easy=200
sciq=200
arc_challenge=400
drop_span=100
gsm8k_zh=100
```

其中各来源含义如下：

| 组件 | 来源 | 样本数 | 任务类型 | 主要答案形式 |
| --- | --- | ---: | --- | --- |
| `arc_easy` | `mib-bench/arc_easy` | 200 | 科学常识选择题 | 选项字母 |
| `sciq` | `allenai/sciq` | 200 | 科学问答 | 短答案/选择式答案 |
| `arc_challenge` | `mib-bench/arc_challenge` | 400 | 挑战性科学选择题 | 选项字母 |
| `drop_span` | `ucinlp/drop` | 100 | 阅读理解 span 抽取 | 文本 span |
| `gsm8k_zh` | `meta-math/GSM8K_zh` 及 verified numeric 补充池 | 100 | 中文数学/数值推理 | 数值答案 |

实现上，当前 v3_1 构建脚本将数值推理组件保存为 `verified_numeric_100.jsonl`。该组件优先从上一轮 v3 baseline 中筛选“小模型错误但大模型或路由正确”的 numeric 样本；不足部分再使用 `GSM8K_zh` numeric 样本补齐。因此在论文或实验说明中可按功能口径称为 `gsm8k_zh=100` 或“数值/中文数学推理 100 条”，但在仓库文件名中对应 `verified_numeric`。

### 2.3 输出路径

组件文件位于：

```text
data/datasets/v3_1_text_fusion/
```

主要文件包括：

```text
arc_easy_500.jsonl
sciq_500.jsonl
arc_challenge_500.jsonl
drop_span_500.jsonl
verified_numeric_100.jsonl
fusion_v3_1_1000_200-200-400-100-100.jsonl
manifest.json
```

最终实验入口文件为：

```text
data/datasets/v3_1_text_fusion/fusion_v3_1_1000_200-200-400-100-100.jsonl
```

### 2.4 字段格式

文本融合数据集的核心字段包括：

| 字段 | 含义 |
| --- | --- |
| `id` | 样本唯一标识 |
| `question` | 输入给 LLM 的问题文本，必要时包含选项或 passage |
| `answer` | 标准答案 |
| `image_path` | 文本任务为空值 `null` |
| `task_type` | 固定为 `text_only` |
| `source` | 原始数据源 |
| `category` | 组件类别，如 `arc_challenge`、`drop_span` |
| `answer_type` | 答案类型，如 `multiple_choice`、`span`、`numeric` |
| `answer_format` | 期望模型输出格式 |
| `choices` / `choice_labels` | 选择题选项及标签 |
| `metadata` | 原始样本编号、筛选桶等附加信息 |

典型选择题样本结构：

```json
{
  "id": "arc_challenge_376",
  "image_path": null,
  "question": "Which item below is NOT made from a material grown in nature?...",
  "answer": "C",
  "task_type": "text_only",
  "source": "mib-bench/arc_challenge",
  "category": "arc_challenge",
  "answer_type": "multiple_choice",
  "choices": ["a cotton shirt", "a wooden chair", "a plastic spoon", "a grass basket"],
  "choice_labels": ["A", "B", "C", "D"]
}
```

### 2.5 构建命令

默认构建命令：

```bash
python scripts/v3_1_prepare_fusion_dataset.py
```

该命令会使用 Hugging Face streaming 读取组件数据，并在本地输出组件 JSONL、融合 JSONL 和 `manifest.json`。如需要只查看参数：

```bash
python scripts/v3_1_prepare_fusion_dataset.py --help
```

## 3. 图像问答数据集

### 3.1 构建目标

图像数据集使用 VQAv2 yes/no 子集，目标是构造适合多模态系统评估的图像问答任务。该数据集主要用于：

- 验证 VLM 是否能从图片生成可复用 caption；
- 测试 image-level caption cache 是否降低重复 VLM 调用；
- 比较 small / medium / large 模型与路由策略的表现；
- 评估 fallback 在多模态问答中的收益和成本。

### 3.2 数据来源

当前图像数据集来自：

```text
lmms-lab/VQAv2-FewShot
config: eval
split: validation
answer_type: yes/no
```

仓库当前使用 1000 条 yes/no 图像问答样本：

```text
data/datasets/vqav2_yesno_1000.jsonl
data/inputs/vqav2_yesno/
```

其中 JSONL 保存问题、答案和图片路径；图片文件保存到 `data/inputs/vqav2_yesno/`。

### 3.3 字段格式

图像问答数据集的核心字段包括：

| 字段 | 含义 |
| --- | --- |
| `id` | 样本唯一标识 |
| `image_path` | 本地图片路径 |
| `question` | 图像相关问题 |
| `answer` | 标准答案，当前为 `yes` 或 `no` |
| `multiple_choice_answer` | VQAv2 原始多数答案 |
| `answers` | VQAv2 标注答案列表 |
| `question_type` | VQAv2 问题类型 |
| `answer_type` | 固定筛选为 `yes/no` |
| `category` | 当前为 `yes_no` |
| `task_type` | 固定为 `image_qa` |
| `source` | 当前为 `VQAv2` |
| `image_id` / `question_id` | VQAv2 原始编号 |

典型样本结构：

```json
{
  "id": "vqav2_393225001",
  "image_path": "data/inputs/vqav2_yesno/393225001.jpg",
  "question": "Is this a creamy soup?",
  "answer": "no",
  "answer_type": "yes/no",
  "category": "yes_no",
  "task_type": "image_qa",
  "source": "VQAv2",
  "image_id": 393225,
  "question_id": 393225001
}
```

### 3.4 构建命令

默认构建命令：

```bash
python scripts/prepare_vqav2_yesno.py \
  --source lmms-lab/VQAv2-FewShot \
  --config-name eval \
  --split validation \
  --limit 1000 \
  --output data/datasets/vqav2_yesno_1000.jsonl \
  --image-dir data/inputs/vqav2_yesno
```

该脚本会：

1. 使用 Hugging Face `datasets` streaming 读取 VQAv2；
2. 默认筛选 `answer_type == "yes/no"` 的样本；
3. 将图片保存到 `data/inputs/vqav2_yesno/`；
4. 生成 AdaRoute-MM 可直接读取的 JSONL；
5. 在已有输出存在时跳过已生成样本，支持断点续跑。

如需强制重建，显式加入：

```bash
--overwrite
```

## 4. 实验使用边界

文本融合数据集和图像问答数据集都服务于 AdaRoute-MM 的路由实验，而不是完整 benchmark 排行榜复现。因此使用时应注意：

- 文本融合数据集的结论适合描述“混合任务下的路由、时延、成本和准确率权衡”，不应直接表述为某个单一 benchmark 的完整性能；
- `gsm8k_zh=100` 在当前文件结构中对应数值推理组件 `verified_numeric_100.jsonl`，其中包含 verified numeric 样本和少量 GSM8K_zh filler；
- VQAv2 图像数据集当前限制在 yes/no 问答，适合稳定评估图像 caption 与路由，不代表开放式 VQA 全量分布；
- 两类数据集均已转换为 AdaRoute-MM runner 可直接读取的 JSONL，可分别用于 text-only suite 和 multimodal suite。
