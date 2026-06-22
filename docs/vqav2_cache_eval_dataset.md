# VQAv2 Cache Evaluation Subsets

This note documents the VQAv2 subsets prepared for Chapter 6 cache analysis.

## Feasibility

The source file is:

```text
data/datasets/vqav2_yesno_1000.jsonl
```

The 1000-row source contains 381 unique image IDs. Among them:

- 120 images have at least 3 questions.
- 73 images have at least 4 questions.
- 47 images have exactly 3 questions.
- 24 images have exactly 4 questions.

Therefore, a clean `100 images x 3 questions = 300 samples` subset is feasible. A strict `100 images x 4 questions = 400 samples` subset is not feasible because only 73 images have at least 4 questions. An average-4-question subset is feasible if some images contribute 3 questions and others contribute 4 or 5.

## Generated Files

The generated subset directory is:

```text
data/datasets/vqav2_cache_eval/
```

Files:

```text
data/datasets/vqav2_cache_eval/vqav2_cache_100img_3q_300.jsonl
data/datasets/vqav2_cache_eval/vqav2_cache_100img_avg4q_400.jsonl
data/datasets/vqav2_cache_eval/manifest.json
data/inputs/vqav2_cache_eval/
```

The script used to build them is:

```text
scripts/build_vqav2_cache_subset.py
```

The cache-on runtime config is:

```text
configs/vqav2_cache_on.yaml
```

## Primary Subset

Use this subset as the main Chapter 6 cache workload:

```text
data/datasets/vqav2_cache_eval/vqav2_cache_100img_3q_300.jsonl
```

Properties:

- 300 rows.
- 100 unique images.
- Exactly 3 questions per image.
- 200 repeated-image rows.
- Theoretical image-level cache hit rate: 66.7%.
- Answer distribution: 172 `no`, 128 `yes`.
- Mean VQAv2 answer consensus: 96.7%.

This subset is the cleanest paper-facing choice because every image contributes the same number of questions. The cache effect can be described as image-level reuse under a controlled multi-question VQA workload.

## Optional Larger Subset

Use this only as a sensitivity or appendix workload:

```text
data/datasets/vqav2_cache_eval/vqav2_cache_100img_avg4q_400.jsonl
```

Properties:

- 400 rows.
- 100 unique images.
- 3 to 5 questions per image.
- Mean 4 questions per image.
- 300 repeated-image rows.
- Theoretical image-level cache hit rate: 75.0%.
- Answer distribution: 216 `no`, 184 `yes`.
- Mean VQAv2 answer consensus: 94.6%.

This subset gives a stronger cache signal, but the per-image question count is uneven. It should not replace the balanced 300-row subset unless the paper explicitly states that the workload is average-balanced rather than exactly balanced.

## Cache-Key Correction

The original VQAv2 preparation saved images by question ID, for example:

```text
data/inputs/vqav2_yesno/262162000.jpg
data/inputs/vqav2_yesno/262162004.jpg
```

Those files can contain the same image content but have different paths. The current VLM cache key uses the resolved image path, file size, file mtime, prompt version, and model name in `image_caption` mode. If the subset keeps question-ID image paths, repeated image IDs may not produce cache hits.

The generated cache-evaluation subsets rewrite `image_path` to one canonical file per `image_id`:

```text
data/inputs/vqav2_cache_eval/<image_id>.jpg
```

This makes the experiment measure real image-level caption reuse.

## Recommended Chapter 6 Metrics

Run the primary 300-row subset once with VLM cache enabled:

- cache on: VLM cache enabled with an empty cache directory at the start of the run.
- cache off: estimate later from the cache-on log by charging repeated-image rows the first observed uncached VLM latency for the same image ID.

Recommended command:

```powershell
.\scripts\run_vqav2_cache_eval.ps1
```

Report:

- end-to-end average latency, P50 latency, and P95 latency;
- VLM-stage average latency;
- VLM latency as a percentage of end-to-end latency;
- image-level cache hit rate;
- accuracy and failure rate, mainly as guardrail metrics.

The paper-facing interpretation should be:

> On a controlled VQAv2 multi-question workload with 100 images and three questions per image, image-level VLM caption caching reduces repeated visual-context extraction cost while preserving the same downstream task distribution.
