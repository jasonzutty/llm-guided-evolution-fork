"""
LLMGE Benchmark Tests — Mutation Quality Comparison

Compares new LLM mutations against saved baselines to track whether
model quality improves or regresses over time.

Tests always PASS — they report results via print output.

Requires: LLM server running, mutated_baselines.csv generated.

Run:
    uv run pytest tests/test_benchmarks.py -v -s
"""
import os
import ast
import csv
import subprocess
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOTA = os.path.join(ROOT, 'sota', 'Titanic')
EVAL_SCRIPT = os.path.join(SOTA, 'eval.py')
INDIVIDUALS_DIR = os.path.join(os.path.dirname(__file__), 'fixtures', 'individuals')
RESULTS_DIR = os.path.join(SOTA, 'results')

BASELINES_CSV = os.path.join(os.path.dirname(__file__), 'fixtures', 'mutated_baselines.csv')

# Individuals that should pass eval.py
VALID_INDIVIDUALS = [
    'xXx04jaaxCjDADdZfQejz3gbNFD',
    'xXx05DUkH2xBIGogZRAFJ8nnJHE',
    'xXx07Gk5iNRGngQLpVQwbMDAAS1',
    'xXx09ejihbvx7yUX9JYxtIVPGDs',
    'xXx0LFCBTNE4Wx7KUjfsG1nZc4I',
    'xXx0MtgG8LTm4mSVWscg4s97g0W',
    'xXx0OCDKQ3u88o9xhlAyHpEcfhM',
    'xXx0T6XRn6JDQPVLadZ4zBFlTd0',
    'xXx0UVvEW16rb3wrfuDSAemYblI',
]


def _load_baselines():
    """Load saved baselines from CSV. Returns dict of gene_id -> {valid_syntax, fp, fn}."""
    baselines = {}
    if not os.path.isfile(BASELINES_CSV):
        return baselines
    with open(BASELINES_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            gene_id = row['input_model'].replace('model_', '')
            entry = {'valid_syntax': row['valid_syntax'] == 'PASS'}
            if row['FP'] != 'ERR':
                entry['fp'] = float(row['FP'])
                entry['fn'] = float(row['FN'])
            else:
                entry['fp'] = None
                entry['fn'] = None
            baselines[gene_id] = entry
    return baselines


def _check_syntax(filepath):
    """Check if file is valid Python with Model(fit, predict)."""
    try:
        with open(filepath) as f:
            tree = ast.parse(f.read())
    except SyntaxError:
        return False, 'invalid Python'

    classes = [n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == 'Model']
    if not classes:
        return False, 'no Model class'

    methods = {n.name for n in ast.walk(classes[0])
               if isinstance(n, ast.FunctionDef)}
    missing = []
    if 'fit' not in methods: missing.append('fit()')
    if 'predict' not in methods: missing.append('predict()')
    if missing:
        return False, f"missing {', '.join(missing)}"

    return True, 'valid'


# ── 1. Mutation benchmark ─────────────────────────────────────────────────

@pytest.mark.parametrize('gene_id', VALID_INDIVIDUALS)
def test_mutation_benchmark(gene_id, tmp_path):
    """Mutate, check syntax, evaluate, compare against saved baseline.

    This test always PASSES. Use -s to see comparison output.
    """
    baselines = _load_baselines()
    if gene_id not in baselines:
        pytest.skip(f"{gene_id}: no baseline in CSV")

    baseline = baselines[gene_id]
    prompt = os.path.join(ROOT, 'tests', 'operators', 'test_prompt_1.txt')
    if not os.path.isfile(prompt):
        pytest.skip("test_prompt_1.txt not found")

    # ── Step 1: Mutate ──
    parent = os.path.join(INDIVIDUALS_DIR, f'model_{gene_id}.py')
    child = os.path.join(str(tmp_path), f'mutated_{gene_id}.py')

    mut_result = subprocess.run(
        ['uv', 'run', 'python', 'src/llm_mutation.py',
         parent, child, prompt,
         '--top_p', '0.1', '--temperature', '0.17',
         '--apply_quality_control', 'False'],
        cwd=ROOT, capture_output=True, text=True, timeout=10000
    )

    if mut_result.returncode != 0 or not os.path.isfile(child):
        print(f"\n  {gene_id}: mutation pipeline failed -- cannot benchmark")
        return

    # ── Step 2: Check syntax ──
    new_syntax_ok, syntax_detail = _check_syntax(child)
    base_syntax_ok = baseline['valid_syntax']

    if base_syntax_ok and new_syntax_ok:
        syntax_verdict = "SAME (both valid)"
    elif not base_syntax_ok and new_syntax_ok:
        syntax_verdict = "IMPROVED (was invalid, now valid)"
    elif base_syntax_ok and not new_syntax_ok:
        syntax_verdict = f"REGRESSED (was valid, now {syntax_detail})"
    else:
        syntax_verdict = f"SAME (both invalid: {syntax_detail})"

    print(f"\n  {gene_id} SYNTAX: baseline={'PASS' if base_syntax_ok else 'FAIL'} "
          f"new={'PASS' if new_syntax_ok else 'FAIL'} -> {syntax_verdict}")

    # ── Step 3: Evaluate (only if syntax is valid) ──
    if not new_syntax_ok:
        print(f"  {gene_id} EVAL: skipped (invalid syntax)")
        return

    eval_result = subprocess.run(
        ['uv', 'run', 'python', EVAL_SCRIPT,
         '--model', f'mutated_{gene_id}',
         '--variant_dir', str(tmp_path)],
        cwd=ROOT, capture_output=True, text=True, timeout=10000
    )

    if eval_result.returncode != 0:
        if baseline['fp'] is not None:
            print(f"  {gene_id} EVAL: REGRESSED (baseline passed eval, new failed)")
        else:
            print(f"  {gene_id} EVAL: SAME (both fail eval)")
        return

    # ── Step 4: Read new results and compare ──
    results_file = os.path.join(RESULTS_DIR, f'{gene_id}_results.txt')
    if not os.path.isfile(results_file):
        print(f"  {gene_id} EVAL: no results file")
        return

    with open(results_file) as f:
        parts = f.read().strip().split(',')
    new_fp, new_fn = float(parts[0].strip()), float(parts[1].strip())
    new_total = new_fp + new_fn

    if baseline['fp'] is None:
        # Baseline failed eval but new one passed -- improvement!
        print(f"  {gene_id} EVAL: IMPROVED (baseline failed eval, "
              f"new FP={new_fp}, FN={new_fn})")
        return

    base_total = baseline['fp'] + baseline['fn']
    diff = new_total - base_total

    if diff < 0:
        eval_verdict = f"IMPROVED by {abs(diff):.1f} (FP+FN: {base_total:.0f} -> {new_total:.0f})"
    elif diff > 0:
        eval_verdict = f"REGRESSED by {diff:.1f} (FP+FN: {base_total:.0f} -> {new_total:.0f})"
    else:
        eval_verdict = f"SAME (FP+FN: {new_total:.0f})"

    print(f"  {gene_id} EVAL: baseline=({baseline['fp']},{baseline['fn']}) "
          f"new=({new_fp},{new_fn}) -> {eval_verdict}")


# ── 2. Crossover benchmark ────────────────────────────────────────────────

CROSSOVER_BASELINES_CSV = os.path.join(os.path.dirname(__file__), 'fixtures', 'crossover_baselines.csv')
CROSSOVER_SCRIPT = os.path.join(ROOT, 'src', 'llm_crossover.py')

CROSSOVER_PAIRS = [
    ('xXx0MtgG8LTm4mSVWscg4s97g0W', 'xXx09ejihbvx7yUX9JYxtIVPGDs'),
    ('xXx0MtgG8LTm4mSVWscg4s97g0W', 'xXx0UVvEW16rb3wrfuDSAemYblI'),
    ('xXx0LFCBTNE4Wx7KUjfsG1nZc4I', 'xXx07Gk5iNRGngQLpVQwbMDAAS1'),
    ('xXx0LFCBTNE4Wx7KUjfsG1nZc4I', 'xXx09ejihbvx7yUX9JYxtIVPGDs'),
    ('xXx0LFCBTNE4Wx7KUjfsG1nZc4I', 'xXx04jaaxCjDADdZfQejz3gbNFD'),
    # Currently failing pairs — tracking for future improvement
    ('xXx04PqApRQMPw55qx6cZsUfVSi', 'xXx04jaaxCjDADdZfQejz3gbNFD'),
    ('xXx04PqApRQMPw55qx6cZsUfVSi', 'xXx05DUkH2xBIGogZRAFJ8nnJHE'),
]


def _load_crossover_baselines():
    """Load crossover baselines from CSV. Returns dict of (x,y) -> {valid_syntax, fp, fn}."""
    baselines = {}
    if not os.path.isfile(CROSSOVER_BASELINES_CSV):
        return baselines
    with open(CROSSOVER_BASELINES_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row['gene_id_x'], row['gene_id_y'])
            entry = {'valid_syntax': row['valid_syntax'] == 'PASS'}
            if row['FP'] != 'ERR':
                entry['fp'] = float(row['FP'])
                entry['fn'] = float(row['FN'])
            else:
                entry['fp'] = None
                entry['fn'] = None
            baselines[key] = entry
    return baselines


@pytest.mark.parametrize('gene_id_x, gene_id_y', CROSSOVER_PAIRS)
def test_crossover_benchmark(gene_id_x, gene_id_y, tmp_path):
    """Cross two individuals, check syntax, evaluate, compare against saved baseline.

    This test always PASSES. Use -s to see comparison output.
    """
    baselines = _load_crossover_baselines()
    pair_key = (gene_id_x, gene_id_y)
    if pair_key not in baselines:
        pytest.skip(f"({gene_id_x}, {gene_id_y}): no baseline in CSV")

    baseline = baselines[pair_key]

    # ── Step 1: Crossover ──
    input_x = os.path.join(INDIVIDUALS_DIR, f'model_{gene_id_x}.py')
    input_y = os.path.join(INDIVIDUALS_DIR, f'model_{gene_id_y}.py')
    crossed_output = os.path.join(str(tmp_path), f'model_crossed_{gene_id_x}_{gene_id_y}.py')

    cross_result = subprocess.run(
        ['uv', 'run', 'python', CROSSOVER_SCRIPT, input_x, input_y, crossed_output],
        cwd=ROOT, capture_output=True, text=True, timeout=10000
    )

    if cross_result.returncode != 0 or not os.path.isfile(crossed_output):
        print(f"\n  ({gene_id_x}, {gene_id_y}): crossover pipeline failed -- cannot benchmark")
        return

    # ── Step 2: Check syntax ──
    new_syntax_ok, syntax_detail = _check_syntax(crossed_output)
    base_syntax_ok = baseline['valid_syntax']

    if base_syntax_ok and new_syntax_ok:
        syntax_verdict = "SAME (both valid)"
    elif not base_syntax_ok and new_syntax_ok:
        syntax_verdict = "IMPROVED (was invalid, now valid)"
    elif base_syntax_ok and not new_syntax_ok:
        syntax_verdict = f"REGRESSED (was valid, now {syntax_detail})"
    else:
        syntax_verdict = f"SAME (both invalid: {syntax_detail})"

    print(f"\n  ({gene_id_x}, {gene_id_y}) SYNTAX: baseline={'PASS' if base_syntax_ok else 'FAIL'} "
          f"new={'PASS' if new_syntax_ok else 'FAIL'} -> {syntax_verdict}")

    # ── Step 3: Evaluate (only if syntax is valid) ──
    if not new_syntax_ok:
        print(f"  ({gene_id_x}, {gene_id_y}) EVAL: skipped (invalid syntax)")
        return

    eval_result = subprocess.run(
        ['uv', 'run', 'python', EVAL_SCRIPT,
         '--model', f'model_crossed_{gene_id_x}_{gene_id_y}',
         '--variant_dir', str(tmp_path)],
        cwd=ROOT, capture_output=True, text=True, timeout=10000
    )

    if eval_result.returncode != 0:
        if baseline['fp'] is not None:
            print(f"  ({gene_id_x}, {gene_id_y}) EVAL: REGRESSED (baseline passed, new failed)")
        else:
            print(f"  ({gene_id_x}, {gene_id_y}) EVAL: SAME (both fail eval)")
        return

    # ── Step 4: Read new results and compare ──
    crossed_gene_id = f'crossed_{gene_id_x}_{gene_id_y}'
    results_file = os.path.join(RESULTS_DIR, f'{crossed_gene_id}_results.txt')
    if not os.path.isfile(results_file):
        print(f"  ({gene_id_x}, {gene_id_y}) EVAL: no results file")
        return

    with open(results_file) as f:
        parts = f.read().strip().split(',')
    new_fp, new_fn = float(parts[0].strip()), float(parts[1].strip())
    new_total = new_fp + new_fn

    if baseline['fp'] is None:
        print(f"  ({gene_id_x}, {gene_id_y}) EVAL: IMPROVED (baseline failed eval, "
              f"new FP={new_fp}, FN={new_fn})")
        return

    base_total = baseline['fp'] + baseline['fn']
    diff = new_total - base_total

    if diff < 0:
        eval_verdict = f"IMPROVED by {abs(diff):.1f} (FP+FN: {base_total:.0f} -> {new_total:.0f})"
    elif diff > 0:
        eval_verdict = f"REGRESSED by {diff:.1f} (FP+FN: {base_total:.0f} -> {new_total:.0f})"
    else:
        eval_verdict = f"SAME (FP+FN: {new_total:.0f})"

    print(f"  ({gene_id_x}, {gene_id_y}) EVAL: baseline=({baseline['fp']},{baseline['fn']}) "
          f"new=({new_fp},{new_fn}) -> {eval_verdict}")
