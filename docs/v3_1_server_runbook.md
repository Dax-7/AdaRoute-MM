# AdaRoute-MM v3_1 服务器运行手册

本手册用于在实验室多人共享 Linux 服务器上隔离运行 v3_1 text-only 实验。建议每个人在自己的目录下 clone 仓库，并使用仓库内 `.venv`，不要共用全局 Python 环境。

## 1. 准备目录

假设你的服务器用户名是 `$USER`，项目放在自己的 `Dax/Projects` 下：

```bash
mkdir -p ~/Dax/Projects
cd ~/Dax/Projects
git clone <你的仓库地址> AdaRoute-MM
cd AdaRoute-MM
```

如果仓库已经 clone 过：

```bash
cd ~/Dax/Projects/AdaRoute-MM
git pull
```

## 2. 创建隔离虚拟环境

推荐 Python 3.10 或 3.11：

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

每次重新登录服务器后，先进入项目并激活环境：

```bash
cd ~/Dax/Projects/AdaRoute-MM
source .venv/bin/activate
```

## 3. 准备 Ollama 模型

确认服务器上 Ollama 服务可用：

```bash
ollama list
```

如果模型不存在，按 `configs/default.yaml` 中的模型名拉取对应模型。例如：

```bash
ollama pull qwen2.5:1.5b
ollama pull phi3:medium
ollama pull gemma2:9b
```

实际模型名以仓库配置为准。如果服务器统一部署了 Ollama，只需要确认当前用户能访问默认的 `http://localhost:11434`。

## 4. 验证代码和数据

先跑静态级别的验证：

```bash
python -m compileall adaroute scripts
python -m pytest tests
```

确认 v3_1 数据集已经在仓库中：

```bash
ls data/datasets/v3_1_text_fusion/
```

关键文件应包括：

```text
fusion_v3_1_1000_200-200-400-100-100.jsonl
manifest.json
verified_numeric_100.jsonl
```

## 5. 运行 v3_1 实验

多人共享服务器上建议使用自己的 run id，避免结果目录互相覆盖：

```bash
RUN_ID="v3_1_${USER}_$(date +%Y%m%d_%H%M%S)"
python scripts/v3_1_run_experiment_suite.py --run-id "$RUN_ID"
```

默认会运行：

```text
always_small
always_gemma
always_middle
difficulty_routing
```

输出目录：

```text
data/experiments_v3_1/<RUN_ID>/
```

`data/experiments_v3_1/` 已被 `.gitignore` 忽略，实验结果默认不会提交到仓库。

## 6. 中断后继续

默认开启 resume。使用同一个 run id 重新执行即可继续：

```bash
python scripts/v3_1_run_experiment_suite.py --run-id "$RUN_ID"
```

如果要从头重跑同一个 run id：

```bash
python scripts/v3_1_run_experiment_suite.py --run-id "$RUN_ID" --no-resume
```

## 7. 单独重算 summary

某个模式跑完后，可以单独重算 summary：

```bash
python scripts/evaluate_results.py \
  --input "data/experiments_v3_1/${RUN_ID}/always_small/results.jsonl" \
  --output "data/experiments_v3_1/${RUN_ID}/always_small/summary.json"
```

## 8. 重新构建 v3_1 数据集

通常不需要在服务器重新构建，因为仓库已经包含 v3_1 fusion 数据。若要调整采样比例或重新生成：

```bash
python scripts/v3_1_prepare_fusion_dataset.py
```

这一步会访问 Hugging Face 数据集；服务器需要能联网，并且可能需要较长时间。
