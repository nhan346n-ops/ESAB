#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

"""Generate AsciiDoc parameter documentation from JSON config files."""


def iter_json_files(conf_dir: Path) -> Iterable[Path]:
    """Yield all .json files under conf_dir recursively."""
    if not conf_dir.exists():
        return
    for p in conf_dir.rglob("*.json"):
        if p.is_file():
            yield p


def load_json(path: Path) -> Dict[str, Any] | None:
    """Load a JSON file, returning None on parse error (with a warning)."""
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"WARN: cant read {path}: {e}", file=sys.stderr)
        return None


def extract_parameters(obj: Any) -> List[Dict[str, Any]]:
    """Extract a list of parameter dicts from a loaded JSON object.

    Handles cases where 'parameters' is missing, not a list, or is a dict.
    Also handles cases where the root of the JSON is a list of parameters.
    """
    if isinstance(obj, list):
        # The JSON file is a list of parameters
        return [p for p in obj if isinstance(p, dict)]

    if not isinstance(obj, dict):
        return []

    params = obj.get("parameters")
    if params is None:
        return []
    if isinstance(params, list):
        return [p for p in params if isinstance(p, dict)]
    if isinstance(params, dict):
        # Some configs might map names->param objects
        return [v for v in params.values() if isinstance(v, dict)]
    return []


def sanitize_cell(text: Any) -> str:
    """Sanitize text for AsciiDoc table cells: escape pipes and flatten newlines."""
    s = "" if text is None else str(text)
    # Escape table separator
    s = s.replace("|", "\\|")
    # Normalize newlines and excessive spaces
    s = " ".join(s.split())
    return s


def build_adoc_for_file(rel_path: Path, params: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append(f"// Generated documentation from source ${rel_path}")
    lines.append("== Parameters")
    lines.append("")
    if not params:
        lines.append("Aucun paramètre trouvé.")
        lines.append("")
    else:
        lines.append('[cols="1,1,3",options="header"]')
        lines.append("|===")
        lines.append("| Name | Type | Description")
        for p in params:
            name = sanitize_cell(p.get("name", ""))
            ptype = sanitize_cell(p.get("type", ""))
            help_ = sanitize_cell(p.get("help", ""))
            lines.append(f"| {name} | {ptype} | {help_}")
        lines.append("|===")
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Generate AsciiDoc parameter documentation from JSON config files.")
    repo_root_default = Path(__file__).resolve().parents[1]
    conf_default = repo_root_default / "src" / "gws" / "conf"
    out_dir_default = Path(__file__).parent / "modules" / "ROOT" / "examples" / "generated"

    parser.add_argument(
        "--conf-dir",
        type=Path,
        default=conf_default,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=out_dir_default,
    )

    args = parser.parse_args(argv)

    conf_dir: Path = args.conf_dir
    out_dir: Path = args.out_dir

    if not conf_dir.exists():
        print(f"ERREUR: dir doest not exist : {conf_dir}", file=sys.stderr)
        return 2

    for json_path in iter_json_files(conf_dir):
        data = load_json(json_path)
        if data is None:
            continue
        params = extract_parameters(data)
        rel_path = json_path.relative_to(conf_dir)
        adoc = build_adoc_for_file(rel_path, params)

        dest_path = out_dir / rel_path.parent / f"{rel_path.stem}_params.adoc"
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_text(adoc, encoding="utf-8")
        print(f"Generated file: {dest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
