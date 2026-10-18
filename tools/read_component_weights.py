"""Inspect verified component tensor groups; this does not construct the fusion model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def load_groups(component: str, weights_dir: Path) -> dict:
    manifest = json.loads((weights_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format_version") != "safir_component_tensors_v1":
        raise ValueError("unsupported component format")
    rows = [r for r in manifest["checkpoints"] if r["id"] == component]
    if len(rows) != 1:
        raise ValueError("unknown or duplicated component ID")
    row = rows[0]
    path = weights_dir / Path(row["file"]).name
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != row["sha256"]:
        raise ValueError("component file SHA256 mismatch")
    from safetensors.torch import load_file
    tensors = load_file(str(path), device="cpu")
    if len(tensors) != row["tensor_count"]:
        raise ValueError("tensor count differs from manifest")
    groups = {}
    consumed = set()
    for mapping in row["state_groups"]:
        name = mapping["original_checkpoint_group"]
        prefix = mapping["exported_key_prefix"]
        if name in groups or not prefix:
            raise ValueError("invalid state-group mapping")
        group = {key[len(prefix):]: value for key, value in tensors.items() if key.startswith(prefix)}
        if not group or "" in group:
            raise ValueError("empty tensor group or tensor key")
        selected = {key for key in tensors if key.startswith(prefix)}
        if consumed.intersection(selected):
            raise ValueError("overlapping state-group prefixes")
        consumed.update(selected)
        groups[name] = group
    if consumed != set(tensors):
        raise ValueError("unmapped tensor keys")
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", required=True)
    parser.add_argument("--weights-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "weights")
    args = parser.parse_args()
    groups = load_groups(args.component, args.weights_dir)
    print(json.dumps({"component": args.component,
                      "groups": {name: len(state) for name, state in groups.items()},
                      "model_constructed": False}, indent=2))


if __name__ == "__main__":
    main()
