"""Tests for privacy-safe detector run provenance."""

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

import run_manifest
from run_manifest import collect_run_manifest, collect_source_metadata, read_text_with_sha256


def _task(results_dir: Path, task_id: int, content: str = "{}") -> Path:
    task_dir = results_dir / f"task_{task_id:03d}"
    workspace = task_dir / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "claude_conversation.json").write_text(content)
    return task_dir


def _collect(task_dirs, **overrides):
    arguments = {
        "benchmark_type": "adr_bench",
        "task_dirs": task_dirs,
        "effective_labels": {"task_001": False, "task_002": True},
        "resolved_concurrency": 7,
        "conversation_hashes": {},
        "artifact_hashes": {
            "config_detector": None,
            "uv_lock": None,
            "tasks": None,
            "agentdojo_ground_truth": None,
        },
        "source": {"commit": None, "dirty": None},
    }
    arguments.update(overrides)
    return collect_run_manifest(**arguments)


def test_hashes_selected_conversations_and_labels_in_task_order(tmp_path: Path):
    results = tmp_path / "results-secret-name"
    second = _task(results, 2, '{"message":"second"}')
    first = _task(results, 1, '{"message":"first"}')
    conversation_hashes = {
        task.name: read_text_with_sha256(task / "workspace" / "claude_conversation.json")[1]
        for task in (second, first)
    }
    conversation_hashes["task_003"] = hashlib.sha256(b"not selected").hexdigest()

    manifest = _collect([second, first], conversation_hashes=conversation_hashes)

    assert manifest["selected_task_ids"] == [1, 2]
    conversations = manifest["inputs"]["conversations"]
    expected = [
        {"task_id": task_id, "sha256": hashlib.sha256(content.encode()).hexdigest()}
        for task_id, content in ((1, '{"message":"first"}'), (2, '{"message":"second"}'))
    ]
    expected_hash = hashlib.sha256(
        json.dumps(expected, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert conversations["aggregate_sha256"] == expected_hash
    assert conversations["count"] == 2
    assert manifest["inputs"]["effective_labels"]["count"] == 2
    assert manifest["resolved_concurrency"] == 7


def test_agentdojo_hashes_actual_ground_truth_file(tmp_path: Path):
    results = tmp_path / "run"
    task = _task(results, 1)
    ground_truth = b'{"task_001":{"is_malicious":true}}'
    ground_truth_path = results / "ground_truth.json"
    ground_truth_path.write_bytes(ground_truth)
    _, ground_truth_hash = read_text_with_sha256(ground_truth_path)

    manifest = _collect(
        [task],
        benchmark_type="agentdojo",
        effective_labels={"task_001": True},
        artifact_hashes={
            "config_detector": None,
            "uv_lock": None,
            "tasks": None,
            "agentdojo_ground_truth": ground_truth_hash,
        },
    )

    artifacts = manifest["inputs"]["artifacts"]
    assert artifacts["agentdojo_ground_truth"] == hashlib.sha256(ground_truth).hexdigest()
    assert artifacts["tasks"] is None


def test_missing_inputs_and_metadata_are_nonfatal_and_explicit(tmp_path: Path, monkeypatch):
    detection_root = tmp_path / "Detection"
    detection_root.mkdir()
    results = tmp_path / "run"
    task = results / "task_001"
    task.mkdir(parents=True)
    monkeypatch.setattr(run_manifest, "_git_output", lambda *args: None)
    source_metadata = collect_source_metadata(detection_root)

    manifest = _collect([task], effective_labels={}, source=source_metadata["source"])

    assert source_metadata == {"source": {"commit": None, "dirty": None}, "uv_lock": None}
    assert manifest["source"] == {"commit": None, "dirty": None}
    assert manifest["inputs"]["conversations"]["missing_task_ids"] == [1]
    assert manifest["inputs"]["effective_labels"]["missing_task_ids"] == [1]
    assert manifest["inputs"]["artifacts"]["config_detector"] is None
    assert manifest["inputs"]["artifacts"]["uv_lock"] is None


def test_git_calls_are_bounded_and_report_clean_or_dirty(tmp_path: Path):
    repo = tmp_path / "repo"
    detection_root = repo / "Detection"
    detection_root.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    (detection_root / "tracked.txt").write_text("clean")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "initial"], check=True)
    clean = collect_source_metadata(detection_root)
    assert clean["source"]["commit"]
    assert clean["source"]["dirty"] is False

    (detection_root / "untracked.txt").write_text("dirty")
    untracked = collect_source_metadata(detection_root)
    assert untracked["source"]["dirty"] is True
    (detection_root / "untracked.txt").unlink()

    (detection_root / "tracked.txt").write_text("dirty")
    dirty = collect_source_metadata(detection_root)
    assert dirty["source"]["dirty"] is True


def test_manifest_excludes_paths_basenames_and_host_identifiers(tmp_path: Path):
    results = tmp_path / "customer-secret-benchmark"
    task = _task(results, 1)

    serialized = json.dumps(_collect([task]))

    assert "customer-secret-benchmark" not in serialized
    assert str(tmp_path) not in serialized
    assert "hostname" not in serialized
    assert "platform" not in serialized
    assert "path" not in serialized


def test_git_commands_have_a_short_timeout(tmp_path: Path, monkeypatch):
    calls = []

    def fake_run(*args, **kwargs):
        calls.append(kwargs)
        return subprocess.CompletedProcess(args[0], 0, stdout="abc123\n", stderr="")

    monkeypatch.setattr(run_manifest.subprocess, "run", fake_run)
    assert run_manifest._git_output(tmp_path, "rev-parse", "HEAD") == "abc123"
    assert calls[0]["timeout"] == run_manifest._GIT_TIMEOUT_SECONDS == 2


def test_git_timeout_is_nonfatal(tmp_path: Path, monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(run_manifest.subprocess, "run", timeout)

    assert run_manifest._git_output(tmp_path, "rev-parse", "HEAD") is None


def test_text_load_hashes_raw_bytes_and_preserves_universal_newlines(tmp_path: Path):
    path = tmp_path / "conversation.json"
    raw = b'{\r\n"message": "caf\xc3\xa9"\r\n}\r'
    path.write_bytes(raw)

    content, digest = read_text_with_sha256(path, encoding="utf-8")

    assert content == '{\n"message": "caf\u00e9"\n}\n'
    assert digest == hashlib.sha256(raw).hexdigest()
    assert digest != hashlib.sha256(content.encode()).hexdigest()


def test_text_and_hash_are_obtained_from_one_file_read(tmp_path: Path, monkeypatch):
    path = tmp_path / "conversation.json"
    raw = b'{"message":"original"}'
    path.write_bytes(raw)
    original_open = Path.open
    calls = []

    def track_open(input_path, *args, **kwargs):
        calls.append(input_path)
        return original_open(input_path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", track_open)

    content, digest = read_text_with_sha256(path, encoding="utf-8")

    assert calls == [path]
    assert content == raw.decode("utf-8")
    assert digest == hashlib.sha256(raw).hexdigest()


def test_hashing_failure_does_not_prevent_text_load(tmp_path: Path, monkeypatch):
    path = tmp_path / "conversation.json"
    path.write_bytes(b'{"message":"still available"}')

    def failed_hash(*args, **kwargs):
        raise RuntimeError("hashing unavailable")

    monkeypatch.setattr(run_manifest.hashlib, "sha256", failed_hash)

    content, digest = read_text_with_sha256(path, encoding="utf-8")

    assert content == '{"message":"still available"}'
    assert digest is None


def test_text_load_still_propagates_file_and_decode_errors(tmp_path: Path):
    path = tmp_path / "conversation.json"
    with pytest.raises(FileNotFoundError):
        read_text_with_sha256(path)

    path.write_bytes(b"\xff")
    with pytest.raises(UnicodeDecodeError):
        read_text_with_sha256(path, encoding="utf-8")


def test_source_metadata_failures_are_nonfatal(tmp_path: Path, monkeypatch):
    def failed_metadata(*args, **kwargs):
        raise RuntimeError("metadata unavailable")

    monkeypatch.setattr(run_manifest, "_git_provenance", failed_metadata)
    monkeypatch.setattr(run_manifest, "_sha256_file", failed_metadata)

    assert collect_source_metadata(tmp_path) == {
        "source": {"commit": None, "dirty": None},
        "uv_lock": None,
    }


def test_final_manifest_uses_captured_inputs_without_rereading(tmp_path: Path, monkeypatch):
    detection_root = tmp_path / "Detection"
    detection_root.mkdir()
    lock = detection_root / "uv.lock"
    lock.write_text("original lock")
    config = detection_root / "config_detector.yaml"
    config.write_text("model: original\n")
    tasks = detection_root / "tasks.json"
    tasks.write_text('{"tasks":[]}')
    task = _task(tmp_path / "results", 1, '{"message":"original"}')
    conversation = task / "workspace" / "claude_conversation.json"
    monkeypatch.setattr(
        run_manifest,
        "_git_output",
        lambda _, *args: "original-commit" if args[0] == "rev-parse" else "",
    )
    source_metadata = collect_source_metadata(detection_root)
    _, conversation_hash = read_text_with_sha256(conversation)
    _, config_hash = read_text_with_sha256(config)
    _, tasks_hash = read_text_with_sha256(tasks)
    captured = {
        "source": source_metadata["source"],
        "conversation_hashes": {task.name: conversation_hash},
        "artifact_hashes": {
            "config_detector": config_hash,
            "uv_lock": source_metadata["uv_lock"],
            "tasks": tasks_hash,
            "agentdojo_ground_truth": None,
        },
    }
    before = _collect([task], **captured)
    conversation.unlink()
    config.write_text("model: replacement\n")
    tasks.write_text('{"tasks":["replacement"]}')
    lock.write_text("replacement lock")

    def unexpected_read(*args, **kwargs):
        raise AssertionError("manifest assembly must not read input files or Git")

    monkeypatch.setattr(Path, "open", unexpected_read)
    monkeypatch.setattr(run_manifest, "_sha256_file", unexpected_read)
    monkeypatch.setattr(run_manifest, "_git_output", unexpected_read)

    after = _collect([task], **captured)

    assert after == before
    assert after["source"] == {"commit": "original-commit", "dirty": False}
    assert after["inputs"]["conversations"]["count"] == 1
    assert after["inputs"]["artifacts"]["uv_lock"] == hashlib.sha256(b"original lock").hexdigest()
