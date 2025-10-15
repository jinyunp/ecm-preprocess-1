from pathlib import Path

from mypkg.core.io import component_paths_from_sanitized


def test_component_paths_without_diagram(tmp_path: Path) -> None:
    sanitized = tmp_path / "processed" / "sample" / "v0" / "_sanitized" / "sample_sanitized.json"
    sanitized.parent.mkdir(parents=True)
    sanitized.write_text("{}", encoding="utf-8")

    paths = component_paths_from_sanitized(sanitized, "sample")
    assert "diagram" not in paths
    expected_keys = {"list", "table", "blocks", "inline_images"}
    assert set(paths.keys()) == expected_keys
