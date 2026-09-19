"""Regression tests for the detector's installed entry point."""

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10; installed with pytest.
    import tomli as tomllib


def test_run_manifest_is_included_in_detector_wheel(detection_root: Path):
    with (detection_root / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)

    force_include = project["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    assert force_include.get("run_manifest.py") == "run_manifest.py"
    assert (detection_root / "run_manifest.py").is_file()
