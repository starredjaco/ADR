"""Tests for main_detector analysis utilities."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from guardrail.base_detector import DetectionResult
from main_detector import BenchmarkAnalyzer, find_latest_benchmark_results, validate_benchmark_results_dir


class _FixedDetector:
    def __init__(self, malicious_tasks: set[str]):
        self._malicious_tasks = malicious_tasks

    def get_info(self):
        return {"name": "FixedDetector"}

    def analyze_task(self, task_data):
        task_id = task_data["task_id"]
        is_malicious = task_id in self._malicious_tasks
        return DetectionResult(
            task_id=task_id,
            is_malicious=is_malicious,
            confidence_score=0.9 if is_malicious else 0.1,
            total_messages=len(task_data.get("messages", [])),
            threat_messages=1 if is_malicious else 0,
            detections=[],
            method="fixed",
            analysis_time=0.5,
            cost_usd=0.01,
        )


class TestAnalyzeTasksRunStats:
    def test_counts_dropped_tasks(self, tmp_path: Path):
        """Tasks returning None (missing conversation) increment dropped, not scored."""
        bench = tmp_path / "adr_bench_test"
        for task_id in ("task_001", "task_002"):
            ws = bench / task_id / "workspace"
            ws.mkdir(parents=True)
            if task_id == "task_001":
                (ws / "claude_conversation.json").write_text(
                    '{"messages": [{"role": "user", "content": "hi"}]}'
                )

        detector = _FixedDetector(set())
        analyzer = BenchmarkAnalyzer(detector)
        ground_truth = {"task_001": False, "task_002": False}

        analyses, run_stats = analyzer._analyze_tasks_efficiently(
            sorted([bench / "task_001", bench / "task_002"]),
            benchmark_type="adr_bench",
            ground_truth_dict=ground_truth,
        )

        assert len(analyses) == 1
        assert run_stats == {"total_tasks": 2, "scored": 1, "dropped": 1}


class TestProcessBenchmarkResults:
    def test_benchmark_type_overrides_dirname_hint(self, tmp_path: Path, capsys):
        """--benchmark agentdojo uses ground_truth.json even when dirname lacks 'agentdojo'."""
        bench = tmp_path / "pinned_run_20251017"
        ws = bench / "task_001" / "workspace"
        ws.mkdir(parents=True)
        (ws / "claude_conversation.json").write_text(
            '{"messages": [{"role": "user", "content": "hi"}]}'
        )
        (bench / "ground_truth.json").write_text(
            json.dumps({"task_001": {"is_malicious": True}})
        )

        detector = _FixedDetector({"task_001"})
        analyzer = BenchmarkAnalyzer(detector)
        result = analyzer.process_benchmark_results(str(bench), benchmark_type="agentdojo")

        captured = capsys.readouterr()
        assert "overrides dirname hint" in captured.out
        assert "Loaded AgentDojo ground truth" in captured.out
        assert result["metrics"]["confusion_matrix"]["true_positives"] == 1
        assert result["metrics"]["confusion_matrix"]["false_negatives"] == 0

    def test_includes_run_stats_in_result_and_json(self, tmp_path: Path):
        """run_stats is returned and survives JSON serialization (saved analysis format)."""
        bench = tmp_path / "adr_bench_test"
        ws = bench / "task_001" / "workspace"
        ws.mkdir(parents=True)
        (ws / "claude_conversation.json").write_text(
            '{"messages": [{"role": "user", "content": "hi"}]}'
        )

        detector = _FixedDetector(set())
        analyzer = BenchmarkAnalyzer(detector)
        result = analyzer.process_benchmark_results(str(bench), benchmark_type="adr_bench")

        assert result["run_stats"] == {"total_tasks": 1, "scored": 1, "dropped": 0}
        saved = json.loads(json.dumps(result))
        assert saved["run_stats"] == {"total_tasks": 1, "scored": 1, "dropped": 0}
        assert saved["run_manifest"]["kind"] == "run_provenance"
        assert saved["run_manifest"]["benchmark_type"] == "adr_bench"
        assert saved["run_manifest"]["selected_task_ids"] == [1]
        assert {"detector_info", "analyses", "metrics", "run_stats", "analysis_timestamp"} <= set(saved)


class TestValidateBenchmarkResultsDir:
    def test_agentdojo_requires_ground_truth_json(self, tmp_path: Path):
        bench = tmp_path / "pinned_run"
        bench.mkdir()
        (bench / "task_001").mkdir()

        with pytest.raises(ValueError, match="requires ground_truth.json"):
            validate_benchmark_results_dir(bench, "agentdojo")

    def test_adr_bench_rejects_ground_truth_json(self, tmp_path: Path):
        bench = tmp_path / "pinned_run"
        bench.mkdir()
        (bench / "ground_truth.json").write_text("{}")
        (bench / "task_001").mkdir()

        with pytest.raises(ValueError, match="does not use per-run ground_truth.json"):
            validate_benchmark_results_dir(bench, "adr_bench")

    def test_adr_bench_accepts_layout_without_ground_truth(self, tmp_path: Path):
        bench = tmp_path / "adr_bench_test"
        bench.mkdir()
        (bench / "task_001").mkdir()

        validate_benchmark_results_dir(bench, "adr_bench")

    def test_agentdojo_accepts_ground_truth_json(self, tmp_path: Path):
        bench = tmp_path / "pinned_run"
        bench.mkdir()
        (bench / "ground_truth.json").write_text("{}")
        (bench / "task_001").mkdir()

        validate_benchmark_results_dir(bench, "agentdojo")


class TestBenchmarkAnalyzerMetrics:
    def test_calculate_metrics_perfect_classifier(self):
        analyzer = BenchmarkAnalyzer(_FixedDetector({"task_002"}))
        analyses = [
            {"task_id": "task_001", "is_malicious": False, "confidence_score": 0.1, "analysis_time": 0.2, "cost_usd": 0.01, "threat_technique": "N/A"},
            {"task_id": "task_002", "is_malicious": True, "confidence_score": 0.9, "analysis_time": 0.3, "cost_usd": 0.02, "threat_technique": "initial_compromise"},
        ]
        ground_truth = {"task_001": False, "task_002": True}

        metrics = analyzer._calculate_metrics(analyses, ground_truth)

        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["confusion_matrix"]["true_positives"] == 1
        assert metrics["confusion_matrix"]["true_negatives"] == 1
        assert metrics["latency_stats"]["total_samples"] == 2
        assert metrics["cost_stats"]["total_usd"] == pytest.approx(0.03)

    def test_calculate_metrics_raises_on_missing_ground_truth(self):
        analyzer = BenchmarkAnalyzer(_FixedDetector(set()))
        analyses = [{"task_id": "task_999", "is_malicious": False, "confidence_score": 0.1}]
        with pytest.raises(ValueError, match="No ground truth"):
            analyzer._calculate_metrics(analyses, {"task_001": False})

    def test_convert_conversation_to_messages(self, sample_conversation):
        analyzer = BenchmarkAnalyzer(MagicMock())
        messages = analyzer._convert_conversation_to_messages(sample_conversation)
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"


class TestFindLatestBenchmarkResults:
    def test_picks_latest_timestamped_directory(self, tmp_path: Path):
        (tmp_path / "adr_bench_20260101_120000").mkdir()
        latest = tmp_path / "adr_bench_20260102_120000"
        latest.mkdir()

        found = find_latest_benchmark_results(tmp_path, "adr_bench")
        assert found == latest

    def test_raises_when_no_results(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="No adr_bench"):
            find_latest_benchmark_results(tmp_path, "adr_bench")

    def test_agentdojo_prefix(self, tmp_path: Path):
        run_dir = tmp_path / "agentdojo_20260103_120000"
        run_dir.mkdir()
        found = find_latest_benchmark_results(tmp_path, "agentdojo")
        assert found == run_dir


class TestGroundTruthLoading:
    def test_load_agentdojo_ground_truth(self, tmp_path: Path):
        analyzer = BenchmarkAnalyzer(MagicMock())
        analyzer.results_dir = tmp_path
        ground_truth_file = tmp_path / "ground_truth.json"
        ground_truth_file.write_text(
            json.dumps({"task_1": {"is_malicious": True}, "task_2": {"is_malicious": False}})
        )

        ground_truth = analyzer._load_ground_truth("agentdojo")
        assert ground_truth["task_001"] is True
        assert ground_truth["task_002"] is False
