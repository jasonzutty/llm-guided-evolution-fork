import subprocess
import ast
import re
from pathlib import Path
import json
import sys

DOC_EXTENSIONS = {".md", ".rst"}
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".tox", ".venv", "venv"}


def _should_skip(path: Path) -> bool:
    return bool(SKIP_DIRS & set(path.parts))


def get_changed_python_files(base_sha: str, head_sha: str) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", base_sha, head_sha],
        capture_output=True, text=True,
    )
    return [
        f for f in result.stdout.strip().split("\n")
        if f.endswith(".py") and Path(f).exists()
    ]


def get_changed_symbols(filepath: str) -> list[str]:
    try:
        tree = ast.parse(Path(filepath).read_text())
    except Exception:
        return []
    symbols: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols.append(node.name)
    return symbols


def check_docstrings(filepath: str) -> list[str]:
    missing: list[str] = []
    try:
        tree = ast.parse(Path(filepath).read_text())
    except Exception:
        return missing
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
            ):
                missing.append(node.name)
    return missing


def _module_name_from_path(filepath: str) -> str:
    return Path(filepath).stem


_RST_DIRECTIVE_RE = re.compile(
    r"""
    (?:
        \.\.\s+auto(?:module|class|function)::\s*(\S+)   # .. automodule:: X
    |
        :py:(?:mod|func|class|meth|attr):`~?([^`]+)`     # :py:mod:`X`
    )
    """,
    re.VERBOSE,
)


def find_doc_references(symbols: list[str], changed_modules: list[str]):
    symbol_hits: list[dict] = []
    directive_hits: list[dict] = []

    for doc_file in Path(".").rglob("*"):
        if doc_file.suffix not in DOC_EXTENSIONS:
            continue
        if _should_skip(doc_file):
            continue

        try:
            content = doc_file.read_text(errors="replace")
        except Exception:
            continue

        matched_syms = [s for s in symbols if s in content]
        if matched_syms:
            symbol_hits.append({
                "doc": str(doc_file),
                "matched_symbols": matched_syms,
            })

        if doc_file.suffix == ".rst" and changed_modules:
            matched_mods: list[str] = []
            for m in _RST_DIRECTIVE_RE.finditer(content):
                ref = m.group(1) or m.group(2)
                if ref is None:
                    continue
                ref_parts = ref.split(".")
                for mod in changed_modules:
                    if mod in ref_parts or mod == ref:
                        matched_mods.append(ref)
                        break

            if matched_mods:
                seen: set[str] = set()
                deduped = []
                for mm in matched_mods:
                    if mm not in seen:
                        seen.add(mm)
                        deduped.append(mm)
                directive_hits.append({
                    "doc": str(doc_file),
                    "matched_directives": deduped,
                })

    return symbol_hits, directive_hits

def run_analysis(base_sha: str, head_sha: str):
    results: list[dict] = []

    changed_py_files = get_changed_python_files(base_sha, head_sha)

    if not changed_py_files:
        print("No changed Python files found.")
        Path("tests/results").mkdir(parents=True, exist_ok=True)
        with open("tests/results/doc_suggestions.json", "w") as fh:
            json.dump([], fh)
        return

    changed_modules = [_module_name_from_path(f) for f in changed_py_files]

    all_symbols: list[str] = []
    for f in changed_py_files:
        symbols = get_changed_symbols(f)
        missing = check_docstrings(f)
        all_symbols.extend(symbols)

        if missing:
            results.append({
                "type": "missing_docstring",
                "file": f,
                "symbols": missing,
                "message": (
                    f"{f} — functions/classes missing docstrings: "
                    + ", ".join(missing)
                ),
            })

    symbol_hits, directive_hits = find_doc_references(
        all_symbols, changed_modules,
    )

    for ref in symbol_hits:
        results.append({
            "type": "doc_reference",
            "file": ref["doc"],
            "symbols": ref["matched_symbols"],
            "message": (
                f"{ref['doc']} — may need updating, references: "
                + ", ".join(ref["matched_symbols"])
            ),
        })

    for ref in directive_hits:
        results.append({
            "type": "rst_directive",
            "file": ref["doc"],
            "symbols": ref["matched_directives"],
            "message": (
                f"{ref['doc']} — Sphinx directive references changed module: "
                + ", ".join(ref["matched_directives"])
            ),
        })


    Path("tests/results").mkdir(parents=True, exist_ok=True)
    with open("tests/results/doc_suggestions.json", "w") as fh:
        json.dump(results, fh, indent=2)

    print(f"\nDoc suggestions ({len(results)} found):")
    for r in results:
        print(f"  [{r['type']}] {r['message']}")

if __name__ == "__main__":
    base_sha = sys.argv[1]
    head_sha = sys.argv[2]
    run_analysis(base_sha, head_sha)