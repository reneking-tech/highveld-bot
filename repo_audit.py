#!/usr/bin/env python3
"""
Highveld Bot Repo Audit
Scans a project directory and produces repo_audit_report.md with key findings.

Usage:
    python repo_audit.py /path/to/highveld_bot
"""

import argparse
import ast
import json
import hashlib
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

MAX_LINE = 120

TRIPLE_QUOTE_PATTERNS = [
    ('"""', '"""'),
    ("""'''""", """'''"""),
]

# Constants we care about in prompt files (reported specially)
PROMPT_CONSTANT_HINTS = {
    "SYSTEM_RAG",
    "STYLE_GUIDE",
    "FIELD_MAPPING_CHEAT_SHEET",
    "QUESTION_UNDERSTANDING",
    "ANSWER_REFINEMENT",
    "QUOTE_PRINCIPLES_BRIEF",
    "PRINCIPLES_BRIEF",
}


def list_files(root: Path) -> Dict[str, List[Path]]:
    files = {"py": [], "json": [], "other": []}
    for p in root.rglob("*"):
        if p.is_file():
            if p.suffix == ".py":
                files["py"].append(p)
            elif p.suffix == ".json":
                files["json"].append(p)
            else:
                files["other"].append(p)
    return files


def check_python_syntax(py_file: Path) -> Tuple[bool, Optional[str]]:
    try:
        src = py_file.read_text(encoding="utf-8")
    except Exception as e:
        return False, f"Could not read file: {e}"
    try:
        ast.parse(src, filename=str(py_file))
        return True, None
    except SyntaxError as e:
        return False, f"SyntaxError: {e.msg} at line {e.lineno}, col {e.offset}"


def extract_all_caps_assignments(py_file: Path) -> Dict[str, Dict[str, Any]]:
    """
    Return mapping CONST_NAME -> {file, lineno, value_hash, sample_preview}
    Only top-level Assign or AnnAssign to Name with ALL_CAPS id.
    """
    results: Dict[str, Dict[str, Any]] = {}
    try:
        src = py_file.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(py_file))
    except Exception:
        return results

    class Visitor(ast.NodeVisitor):
        def visit_Assign(self, node: ast.Assign) -> Any:
            for t in node.targets:
                if isinstance(t, ast.Name) and re.fullmatch(r"[A-Z_][A-Z0-9_]*", t.id):
                    value_src = ast.get_source_segment(src, node.value) or ""
                    h = hashlib.sha256(value_src.encode("utf-8", errors="ignore")).hexdigest()[:12]
                    results.setdefault(t.id, []).append(
                        {
                            "file": str(py_file),
                            "lineno": node.lineno,
                            "hash": h,
                            "preview": value_src[:140].replace("\n", "\\n"),
                        }
                    )

        def visit_AnnAssign(self, node: ast.AnnAssign) -> Any:
            t = node.target
            if isinstance(t, ast.Name) and re.fullmatch(r"[A-Z_][A-Z0-9_]*", t.id):
                value_src = ast.get_source_segment(src, node.value) or ""
                h = hashlib.sha256(value_src.encode("utf-8", errors="ignore")).hexdigest()[:12]
                results.setdefault(t.id, []).append(
                    {
                        "file": str(py_file),
                        "lineno": node.lineno,
                        "hash": h,
                        "preview": value_src[:140].replace("\n", "\\n"),
                    }
                )

    Visitor().visit(tree)
    return results


def check_json(json_file: Path) -> Tuple[bool, Optional[str]]:
    try:
        with json_file.open("r", encoding="utf-8") as f:
            json.load(f)
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def triple_quote_balance_heuristic(text: str) -> Dict[str, int]:
    counts = {}
    counts['"""'] = text.count('"""')
    counts["'''"] = text.count("'''")
    return counts


def collect_long_lines(py_file: Path, max_len: int = MAX_LINE) -> List[Tuple[int, int]]:
    res = []
    try:
        for i, line in enumerate(py_file.read_text(encoding="utf-8").splitlines(), 1):
            if len(line) > max_len:
                res.append((i, len(line)))
    except Exception:
        pass
    return res


def find_todos(py_file: Path) -> List[Tuple[int, str]]:
    res = []
    pattern = re.compile(r"\b(TODO|FIXME|HACK|XXX)\b", re.IGNORECASE)
    try:
        for i, line in enumerate(py_file.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line):
                res.append((i, line.strip()))
    except Exception:
        pass
    return res


def main():
    ap = argparse.ArgumentParser(
        description="Audit a Highveld Bot repo and output a markdown report."
    )
    # Make the path optional; default to the current working directory when omitted
    ap.add_argument(
        "path",
        nargs="?",
        default=".",
        type=str,
        help="Path to the project root (default: current directory)",
    )
    args = ap.parse_args()

    root = Path(args.path).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        print(f"ERROR: Path not found or not a directory: {root}")
        return 2

    files = list_files(root)

    # Syntax checks
    py_ok = []
    py_fail = []
    for f in files["py"]:
        ok, err = check_python_syntax(f)
        if ok:
            py_ok.append(f)
        else:
            py_fail.append((f, err))

    # ALL_CAPS duplicates & prompt constants
    const_map: Dict[str, List[Dict[str, Any]]] = {}
    for f in files["py"]:
        for name, entries in extract_all_caps_assignments(f).items():
            const_map.setdefault(name, []).extend(entries)

    duplicate_consts = {name: entries for name, entries in const_map.items() if len(entries) > 1}
    prompt_consts = {k: v for k, v in const_map.items() if k in PROMPT_CONSTANT_HINTS}

    # JSON validation
    json_ok = []
    json_fail = []
    for jf in files["json"]:
        ok, err = check_json(jf)
        if ok:
            json_ok.append(jf)
        else:
            json_fail.append((jf, err))

    # Triple-quote heuristic + long lines + todos
    triple_issues = []
    long_lines_report = []
    todos_report = []
    for f in files["py"]:
        try:
            txt = f.read_text(encoding="utf-8")
        except Exception:
            continue
        counts = triple_quote_balance_heuristic(txt)
        if (counts['"""'] % 2) != 0 or (counts["'''"] % 2) != 0:
            triple_issues.append(str(f))

        ll = collect_long_lines(f, MAX_LINE)
        if ll:
            # cap to 50 lines per file
            long_lines_report.append((str(f), ll[:50]))

        todos = find_todos(f)
        if todos:
            todos_report.append((str(f), todos[:50]))

    report_lines: List[str] = []
    rl = report_lines.append

    rl("# Highveld Bot – Repository Audit Report")
    rl("")
    rl(f"**Root:** `{root}`")
    rl("")
    rl("## Summary")
    rl(f"- Python files: {len(files['py'])}")
    rl(f"- JSON files: {len(files['json'])}")
    rl(f"- Other files: {len(files['other'])}")
    rl("")
    rl("### Results")
    rl(f"- Python syntax OK: {len(py_ok)}")
    rl(f"- Python syntax errors: {len(py_fail)}")
    rl(f"- JSON valid: {len(json_ok)}")
    rl(f"- JSON errors: {len(json_fail)}")
    rl(f"- Duplicate ALL_CAPS constants: {len(duplicate_consts)}")
    rl(f"- Files with possible triple-quote imbalance: {len(triple_issues)}")
    rl(f"- Files with long lines (> {MAX_LINE} chars): {len(long_lines_report)}")
    rl(f"- Files with TODO/FIXME: {len(todos_report)}")
    rl("")

    if py_fail:
        rl("## Python Syntax Errors")
        for f, err in py_fail:
            rl(f"- `{f}` — {err}")
        rl("")

    if json_fail:
        rl("## JSON Errors")
        for jf, err in json_fail:
            rl(f"- `{jf}` — {err}")
        rl("")

    if duplicate_consts:
        rl("## Duplicate ALL-CAPS Constants")
        for name, entries in sorted(duplicate_consts.items()):
            rl(f"### {name}")
            for e in entries:
                rl(
                    f"- `{e['file']}`:{e['lineno']} — hash `{e['hash']}` — preview: `{e['preview']}`"
                )
            rl("")
    else:
        rl("## Duplicate ALL-CAPS Constants")
        rl("_None detected._")
        rl("")

    if prompt_consts:
        rl("## Prompt Constant Occurrences (FYI)")
        for name, entries in sorted(prompt_consts.items()):
            rl(f"### {name}")
            for e in entries:
                rl(f"- `{e['file']}`:{e['lineno']} — hash `{e['hash']}`")
            rl("")
    else:
        rl("## Prompt Constant Occurrences (FYI)")
        rl("_None detected._")
        rl("")

    if triple_issues:
        rl("## Possible Triple-Quote Imbalance")
        for f in triple_issues:
            rl(f"- `{f}`")
        rl(
            "> Heuristic flag: file contains an odd count of triple quotes. Verify large string blocks."
        )
        rl("")

    if long_lines_report:
        rl("## Long Lines")
        for fname, items in long_lines_report:
            rl(f"### {fname}")
            for lineno, length in items:
                rl(f"- Line {lineno} — {length} chars")
        rl("")

    if todos_report:
        rl("## TODO / FIXME")
        for fname, items in todos_report:
            rl(f"### {fname}")
            for lineno, text in items:
                rl(f"- Line {lineno}: {text}")
        rl("")

    # Write report
    report_path = root / "repo_audit_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Report written to: {report_path}")


if __name__ == "__main__":
    raise SystemExit(main())
