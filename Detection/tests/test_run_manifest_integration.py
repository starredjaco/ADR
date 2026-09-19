"""Run provenance must describe the bytes consumed, not later on-disk state."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

import main_detector
from guardrail.base_detector import DetectionResult
from main_detector import BenchmarkAnalyzer


def _sha256(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


def _conversation_hash(content: str) -> str:
    return _sha256(
        json.dumps(
            [{"task_id": 1, "sha256": _sha256(content)}],
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _inputs(tmp_path: Path, benchmark_type: str):
    bench = tmp_path / f"{benchmark_type}_synthetic"
    workspace = bench / "task_001" / "workspace"
    workspace.mkdir(parents=True)
    conversation = workspace / "claude_conversation.json"
    content = json.dumps([{"role": "user", "content": "original synthetic input"}])
    conversation.write_text(content)
    if benchmark_type == "agentdojo":
        labels = bench / "ground_truth.json"
        label_content = json.dumps({"task_001": {"is_malicious": False}})
    else:
        labels = tmp_path / "tasks.json"
        label_content = json.dumps(
            {"tasks": [{"task_id": 1, "ground_truth": "benign", "mcp_servers": ["original"]}]}
        )
    labels.write_text(label_content)
    return bench, conversation, content, labels, label_content


class _StubDetector:
    def get_info(self):
        return {"name": "StubDetector"}

    def analyze_task(self, task_data):
        return DetectionResult(
            task_id=task_data["task_id"],
            is_malicious=False,
            confidence_score=0.1,
            total_messages=len(task_data["messages"]),
            threat_messages=0,
            detections=[],
            method="synthetic",
            analysis_time=0.01,
        )


@pytest.mark.parametrize("benchmark_type", ["adr_bench", "agentdojo"])
def test_manifest_retains_consumed_inputs_after_files_change(tmp_path, monkeypatch, benchmark_type):
    bench, conversation, content, labels, label_content = _inputs(tmp_path, benchmark_type)
    monkeypatch.chdir(tmp_path)
    source = {"commit": "before-analysis", "dirty": False}
    metadata_calls = []

    def source_metadata(root):
        metadata_calls.append(root)
        return {"source": dict(source), "uv_lock": "lock-before-analysis"}

    monkeypatch.setattr(main_detector, "collect_source_metadata", source_metadata)

    class MutatingDetector(_StubDetector):
        def analyze_task(self, task_data):
            assert task_data["messages"][-1]["content"] == "original synthetic input"
            conversation.write_text("[]")
            labels.unlink()
            source.update(commit="after-analysis", dirty=True)
            return super().analyze_task(task_data)

    result = BenchmarkAnalyzer(MutatingDetector()).process_benchmark_results(
        str(bench), benchmark_type=benchmark_type, max_concurrent=1
    )
    manifest = result["run_manifest"]
    assert result["run_stats"] == {"total_tasks": 1, "scored": 1, "dropped": 0}
    assert manifest["inputs"]["conversations"]["aggregate_sha256"] == _conversation_hash(content)
    artifact = "tasks" if benchmark_type == "adr_bench" else "agentdojo_ground_truth"
    assert manifest["inputs"]["artifacts"][artifact] == _sha256(label_content)
    assert manifest["source"] == {"commit": "before-analysis", "dirty": False}
    assert manifest["inputs"]["artifacts"]["uv_lock"] == "lock-before-analysis"
    assert manifest["inputs"]["artifacts"]["config_detector"] is None
    assert len(metadata_calls) == 1


def test_labels_and_mcp_definitions_share_one_snapshot(tmp_path, monkeypatch):
    bench, _, _, tasks_file, original = _inputs(tmp_path, "adr_bench")
    monkeypatch.chdir(tmp_path)

    class ReplacingAnalyzer(BenchmarkAnalyzer):
        def _load_ground_truth(self, benchmark_type):
            labels = super()._load_ground_truth(benchmark_type)
            tasks_file.write_text(
                json.dumps(
                    {"tasks": [{"task_id": 1, "ground_truth": "malicious", "mcp_servers": ["new"]}]}
                )
            )
            return labels

    class InspectingDetector(_StubDetector):
        def analyze_task(self, task_data):
            assert task_data["messages"][0]["mcp_servers"] == ["original"]
            return super().analyze_task(task_data)

    result = ReplacingAnalyzer(InspectingDetector()).process_benchmark_results(str(bench))
    assert result["run_stats"]["scored"] == 1
    assert result["analyses"][0]["ground_truth_binary"] is False
    assert result["run_manifest"]["inputs"]["artifacts"]["tasks"] == _sha256(original)


def test_reused_analyzer_does_not_retain_previous_conversation_hash(tmp_path, monkeypatch):
    bench, conversation, _, _, _ = _inputs(tmp_path, "adr_bench")
    monkeypatch.chdir(tmp_path)
    analyzer = BenchmarkAnalyzer(_StubDetector())
    first = analyzer.process_benchmark_results(str(bench))
    conversation.unlink()
    second = analyzer.process_benchmark_results(str(bench))
    assert first["run_manifest"]["inputs"]["conversations"]["count"] == 1
    assert second["run_manifest"]["inputs"]["conversations"]["count"] == 0
    assert second["run_manifest"]["inputs"]["conversations"]["missing_task_ids"] == [1]


def test_malformed_conversation_retains_attempted_input_hash(tmp_path, monkeypatch):
    bench, conversation, _, _, _ = _inputs(tmp_path, "adr_bench")
    monkeypatch.chdir(tmp_path)
    malformed = '{"messages": ['
    conversation.write_text(malformed)
    result = BenchmarkAnalyzer(_StubDetector()).process_benchmark_results(str(bench))
    assert result["run_stats"] == {"total_tasks": 1, "scored": 0, "dropped": 1}
    provenance = result["run_manifest"]["inputs"]["conversations"]
    assert provenance["aggregate_sha256"] == _conversation_hash(malformed)
    assert provenance["missing_task_ids"] == []


def test_cli_hashes_consumed_working_directory_config_and_tasks(tmp_path, monkeypatch):
    bench, _, _, tasks_file, _ = _inputs(tmp_path, "adr_bench")
    config_file = tmp_path / "config_detector.yaml"
    observed = {}

    class ConfiguredDetector(_StubDetector):
        def __init__(self, model_name, **kwargs):
            observed["model"] = model_name
            # Model setup may take time: later edits must not replace the hash.
            config_file.write_text("llamafirewall:\n  model: unconsumed-replacement\n")

        def is_available(self):
            return True

        def analyze_task(self, task_data):
            observed["mcp_servers"] = task_data["messages"][0]["mcp_servers"]
            return super().analyze_task(task_data)

    monkeypatch.setattr(main_detector, "LlamaFirewallBaseline", ConfiguredDetector)
    monkeypatch.setattr(main_detector, "print_analysis_summary", lambda analysis: None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["main_detector.py", "--detector", "llamafirewall", "--results-dir", str(bench)],
    )
    monkeypatch.chdir(tmp_path)
    manifests = []
    for version in ("first", "second"):
        config = f"llamafirewall:\n  model: synthetic-{version}\n"
        tasks = json.dumps(
            {"tasks": [{"task_id": 1, "ground_truth": "benign", "mcp_servers": [version]}]}
        )
        config_file.write_text(config)
        tasks_file.write_text(tasks)
        main_detector.main()
        assert observed == {"model": f"synthetic-{version}", "mcp_servers": [version]}
        saved = json.loads((bench / "llamafirewall_baseline_analysis.json").read_text())
        assert saved["run_stats"]["scored"] == 1
        manifest = saved["run_manifest"]
        assert manifest["inputs"]["artifacts"]["config_detector"] == _sha256(config)
        assert manifest["inputs"]["artifacts"]["tasks"] == _sha256(tasks)
        manifests.append(manifest)
    assert manifests[0]["inputs"]["artifacts"] != manifests[1]["inputs"]["artifacts"]
