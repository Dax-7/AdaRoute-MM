# AdaRoute-MM 音频模态扩展规划

## 扩展定位

本规划用于说明 AdaRoute-MM 在系统架构层面对音频模态的可扩展性。当前音频分支仅作为 architecture-level extension，用于论文或答辩中的多模态扩展表达，不纳入本文主实验结果，不改变已有文本融合数据集、路由逻辑或实验统计。

## 音频模型选择

音频分支建议采用 Whisper tiny 作为 ASR 前端。选择该模型的原因是：

- 模型规模小，符合 AdaRoute-MM 面向本地、轻量、资源受限部署的系统设定。
- Whisper 系列是通用语音识别模型，接口和任务边界清晰，适合作为音频到文本的预处理模块。
- tiny 版本可以将音频模态接入成本控制在较低水平，避免让音频前端本身成为主系统的主要计算负担。

在架构图中不突出 Whisper tiny 的模型名称，而是将其抽象为音频转写前端：`Audio + Question -> Speech Transcript`。这样可以强调系统接口，而不是引入新的主实验模型对比。

## 数据集示例选择

音频数据集示例采用 `openslr/librispeech_asr`，任务类型为 Automatic Speech Recognition, ASR。选择该数据集的原因是：

- LibriSpeech 是标准语音识别数据集，任务定义稳定，适合说明 speech/audio input 到 transcript 的转换。
- 数据集目标是语音转写，与本扩展分支的作用一致。
- 该数据集不涉及环境声 caption，不会把音频分支误导为 Clotho、AudioCaps 等声景描述任务。

该数据集仅作为架构扩展示例，不作为当前 AdaRoute-MM 主实验数据来源。

## 接入方式

音频分支在现有 pipeline 之前增加一个轻量 ASR 前处理步骤：

1. 输入形式为 `Audio + Question`。
2. ASR 前端将 speech/audio input 转写为 `Speech Transcript`。
3. `Speech Transcript` 与原始问题组合为统一文本上下文。
4. 统一文本上下文进入已有 Difficulty Router。
5. 后续仍沿用现有 routing pipeline，包括难度判断、本地模型选择、硬件状态感知、答案生成与评估。

因此，音频分支不需要改变 AdaRoute-MM 的核心路由接口。它本质上是一个把音频输入规范化为文本上下文的 modality adapter。

## 主实验保持不变

当前主实验仍然使用已有文本融合数据集，并继续围绕文本与图像+文本输入评估 AdaRoute-MM 的路由收益。音频分支暂不进入主实验，原因是：

- 本阶段目标是展示系统架构的多模态扩展能力，而不是引入新的音频 benchmark。
- 新增音频实验会改变实验边界，需要额外的数据构建、ASR 误差分析和跨模态公平性说明。
- 为保持论文主实验口径一致，现有准确率、延迟、token cost 和大模型调用比例等结果不应因架构扩展图而变化。

后续如果需要实证验证音频分支，应作为独立扩展实验设计，单独记录数据来源、ASR 模型配置、转写质量影响和路由评估结果。
