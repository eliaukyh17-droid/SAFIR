"""Verify archive bytes; never import the model, deserialize weights, or download files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys


def check_manifest(manifest: dict) -> list[dict]:
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported checkpoint manifest version")
    records = manifest.get("checkpoints")
    if not isinstance(records, list) or not records:
        raise ValueError("checkpoint list is empty or invalid")
    seen = set()
    for record in records:
        path = record.get("file", "")
        parts = PurePosixPath(path).parts
        if (not path or "\\" in path or ":" in path or not parts
                or PurePosixPath(path).is_absolute() or ".." in parts
                or parts[0] != "weights" or path in seen):
            raise ValueError("checkpoint paths must be unique relative weights/ paths")
        seen.add(path)
        if not re.fullmatch(r"[0-9a-f]{64}", record.get("sha256", "")):
            raise ValueError("invalid SHA256 in checkpoint inventory")
    return records


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(root: Path, records: list[dict]) -> list[dict]:
    root = root.resolve()
    results = []
    for record in records:
        relative = record["file"]
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("checkpoint resolves outside the supplied root")
        result = {"id": record["id"], "file": relative,
                  "expected_sha256": record["sha256"]}
        if not path.is_file():
            result["status"] = "missing"
        else:
            result["size_bytes"] = path.stat().st_size
            result["actual_sha256"] = file_sha256(path)
            result["status"] = ("match" if result["actual_sha256"] == record["sha256"]
                                else "mismatch")
        results.append(result)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."),
                        help="directory containing the weights/ asset directory")
    parser.add_argument("--manifest", type=Path,
                        default=Path(__file__).resolve().parents[1] / "weights/manifest.json")
    args = parser.parse_args()
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        results = verify(args.root, check_manifest(manifest))
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Verification failed: {error}", file=sys.stderr)
        return 2
    report = {"all_match": all(item["status"] == "match" for item in results),
              "scope": "raw archived file identity only; no model execution",
              "checkpoints": results}
    print(json.dumps(report, indent=2))
    return 0 if report["all_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
