# 服务器部署与批量实验指南

这份指南用于在 Linux 云服务器上 clone AdaRoute-MM，并运行 v2 版 VQAv2 yes/no 批量实验。

## 1. 克隆仓库

```bash
git clone <your-repo-url> AdaRoute-MM
cd AdaRoute-MM
```

仓库里应该已经包含服务器实验所需的派生数据：

```text
data/datasets/vqav2_yesno_1000.jsonl
data/inputs/vqav2_yesno/*.jpg
```

原始 VQAv2 parquet 文件不需要提交到仓库，也不需要放到服务器上才能跑这批实验。

## 2. 创建 Python 环境

建议使用 Python 3.10 或 3.11。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pytest
```

## 3. 安装并启动 Ollama

在 Linux 上安装 Ollama：

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

启动 Ollama：

```bash
ollama serve
```

如果服务器使用 systemd，安装后 Ollama 可能已经作为服务运行，可以这样检查：

```bash
systemctl status ollama
```

## 4. 拉取实验需要的模型

`configs/default.yaml` 默认使用以下 Ollama 模型名：

```bash
ollama pull moondream
ollama pull qwen2.5:1.5b-instruct-q4_1
ollama pull qwen2.5:1.5b
ollama pull phi3
ollama pull sam860/gemma3n:e2b-Q3_K_XL
```

拉取完成后检查模型是否齐全：

```bash
python scripts/check_ollama_models.py --config configs/default.yaml
```

## 5. 做一次 Linux 兼容性检查

先跑测试，确认环境和基础逻辑正常：

```bash
python -m pytest tests
```

当前项目主体使用 `pathlib.Path` 处理路径，准备数据里的 `image_path` 也保存为正斜杠相对路径，因此 clone 到 Linux 后应该可以直接运行。注意不要把数据文件移出仓库根目录，否则需要同步修改 JSONL 里的 `image_path`。

## 6. 检查准备好的数据

确认 JSONL 和图片都在：

```bash
test -f data/datasets/vqav2_yesno_1000.jsonl
test -d data/inputs/vqav2_yesno
find data/inputs/vqav2_yesno -maxdepth 1 -type f -name '*.jpg' | wc -l
```

最后一条命令应输出 `1000`。

## 7. 先跑一个单模式 smoke test

建议先跑一个模式，确认 Ollama、模型、数据路径都没问题：

```bash
python scripts/run_experiment.py \
  --mode difficulty_cache \
  --dataset data/datasets/vqav2_yesno_1000.jsonl \
  --run-id vqav2_yesno_server_smoke
```

输出目录：

```text
data/experiments/vqav2_yesno_server_smoke/difficulty_cache/
```

重点看这几个文件：

```text
results.jsonl
summary.json
resolved_config.yaml
```

## 8. 跑完整 v2 实验套件

smoke test 正常后，运行完整 ablation suite：

```bash
python scripts/run_experiment_suite.py \
  --suite vqav2_yesno_ablation \
  --dataset data/datasets/vqav2_yesno_1000.jsonl \
  --run-id vqav2_yesno_v2_server
```

当前 suite 会依次运行：

```text
always_small
always_gemma
random_routing
difficulty_routing
difficulty_cache
difficulty_fallback
latency_aware_routing
adaroute_mm_full
```

每个模式都会在 `data/experiments/<run-id>/<mode>/` 下生成独立的 `results.jsonl` 和 `summary.json`。

## 9. 中断后续跑或重新跑

默认开启 resume。只要使用同一个 `--run-id` 再跑同一条命令，就会从已有的 `results.jsonl` 继续。

如果要强制重新跑，换一个新的 `--run-id`，或者加上 `--no-resume`：

```bash
python scripts/run_experiment.py \
  --mode difficulty_cache \
  --dataset data/datasets/vqav2_yesno_1000.jsonl \
  --run-id vqav2_yesno_fresh \
  --no-resume
```

## 10. 重新生成 summary

如果某个结果 JSONL 已经存在，只想重新算 summary：

```bash
python scripts/evaluate_yesno_results.py \
  --input data/experiments/vqav2_yesno_v2_server/difficulty_cache/results.jsonl \
  --output data/experiments/vqav2_yesno_v2_server/difficulty_cache/summary.json
```

## 11. 服务器跑完后提交结果

`.gitignore` 已经保留这些可提交文件：

```text
data/datasets/vqav2_yesno_1000.jsonl
data/inputs/vqav2_yesno/*.jpg
data/experiments/**/results.jsonl
data/experiments/**/summary.json
data/experiments/**/resolved_config.yaml
```

同时继续忽略运行缓存、日志、每条请求的中间 JSON 和原始大数据文件。

服务器实验结束后提交：

```bash
git status --short
git add data/datasets/vqav2_yesno_1000.jsonl data/inputs/vqav2_yesno data/experiments
git commit -m "Add VQAv2 yes/no experiment data and results"
git push
```

