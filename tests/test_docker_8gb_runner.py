import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCKER_SCRIPTS = ROOT / "docker" / "scripts"
if str(DOCKER_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(DOCKER_SCRIPTS))

from run_docker_8gb_suite import write_stratified_sample


def test_write_stratified_sample_preserves_source_ratio(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    rows = []
    for source, count in [("a", 200), ("b", 300), ("c", 500)]:
        for index in range(count):
            rows.append({"id": f"{source}-{index}", "source": source, "question": "q", "answer": "a"})
    dataset.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    output = tmp_path / "sample.jsonl"
    manifest = write_stratified_sample(dataset, output, sample_size=100, stratify_by="source")
    sampled = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    distribution = {}
    for row in sampled:
        distribution[row["source"]] = distribution.get(row["source"], 0) + 1

    assert manifest["sample_count"] == 100
    assert distribution == {"a": 20, "b": 30, "c": 50}
    assert set(sampled[0]) == {"id", "source", "question", "answer"}
