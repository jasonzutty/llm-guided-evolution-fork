"""
LLMGE Individual Evaluation Tests

Tests eval.py on 12 frozen individuals from a real LLMGE run.
Each individual is evaluated and checked for valid output.

No LLM server needed — just evaluates pre-generated models.

Run:
    uv run pytest tests/test_individuals.py -v
"""
import os
import ast
import subprocess
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOTA = os.path.join(ROOT, 'sota', 'Titanic')
EVAL_SCRIPT = os.path.join(SOTA, 'eval.py')
CROSSOVER_SCRIPT = os.path.join(ROOT, 'src', 'llm_crossover.py')
INDIVIDUALS_DIR = os.path.join(os.path.dirname(__file__), 'fixtures', 'individuals')
RESULTS_DIR = os.path.join(SOTA, 'results')

# Total samples in the validation set (used for upper bound check)
MAX_SAMPLES = 179  # based on processed_train.csv 80/20 split

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

# Individuals expected to fail eval.py (kept to verify eval still catches them)
INVALID_INDIVIDUALS = {
    'xXx061oJQctwH2mBMMPeIwd7sCk': 'SyntaxError',   # missing bracket
    'xXx04PqApRQMPw55qx6cZsUfVSi': 'TypeError',       # keywords must be strings
    'xXx0WcxFNILJIkWzWO9MsEGL0Jx': 'TypeError',      # int instead of list in GridSearchCV
}

ALL_INDIVIDUALS = VALID_INDIVIDUALS + list(INVALID_INDIVIDUALS.keys())

# Individuals that produce valid Python after mutation (subset of VALID_INDIVIDUALS)
VALID_MUTATE = [
    'xXx04jaaxCjDADdZfQejz3gbNFD',
    'xXx07Gk5iNRGngQLpVQwbMDAAS1',
    'xXx09ejihbvx7yUX9JYxtIVPGDs',
    'xXx0LFCBTNE4Wx7KUjfsG1nZc4I',
    'xXx0OCDKQ3u88o9xhlAyHpEcfhM',
]


# ── 1. Each individual is valid Python with a Model class. Ensures models haven't changed and all can be evaluated. ──────────────────

@pytest.mark.parametrize('gene_id', VALID_INDIVIDUALS)
def test_individual_is_valid_python(gene_id):
    """Frozen individual parses as Python with Model(fit, predict)."""
    path = os.path.join(INDIVIDUALS_DIR, f'model_{gene_id}.py')
    with open(path) as f:
        tree = ast.parse(f.read())
    classes = [n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == 'Model']
    assert len(classes) >= 1, f"{gene_id}: no Model class"
    methods = {n.name for n in ast.walk(classes[0])
               if isinstance(n, ast.FunctionDef)}
    assert 'fit' in methods, f"{gene_id}: Model missing fit()"
    assert 'predict' in methods, f"{gene_id}: Model missing predict()"


# ── 2. Valid individuals evaluate and produce valid results ────────────────

@pytest.mark.parametrize('gene_id', VALID_INDIVIDUALS)
def test_individual_evaluates(gene_id):
    """eval.py runs on the individual and produces bounded FP,FN."""
    model_name = f'model_{gene_id}'

    result = subprocess.run(
        ['uv', 'run', 'python', EVAL_SCRIPT,
         '--model', model_name,
         '--variant_dir', INDIVIDUALS_DIR],
        cwd=ROOT, capture_output=True, text=True, timeout=10000 # need to find more exact timeout value for different datasets this is going to be run on
    )

    assert 'job done' in result.stdout.lower(), \
        f"{gene_id}: 'job done' not in output"

    # Check results file was written
    results_file = os.path.join(RESULTS_DIR, f'{gene_id}_results.txt')
    assert os.path.isfile(results_file), \
        f"{gene_id}: results file not created"

    # Validate format and bounds
    with open(results_file) as f:
        content = f.read().strip()
    parts = content.split(',')
    assert len(parts) == 2, f"{gene_id}: expected 'FP,FN', got '{content}'"

    fp, fn = float(parts[0].strip()), float(parts[1].strip())
    assert 0 <= fp <= MAX_SAMPLES, f"{gene_id}: FP={fp} out of bounds [0, {MAX_SAMPLES}]"
    assert 0 <= fn <= MAX_SAMPLES, f"{gene_id}: FN={fn} out of bounds [0, {MAX_SAMPLES}]"


# ── 3. Known-bad individuals should fail eval.py ──────────────────────────

@pytest.mark.parametrize('gene_id', INVALID_INDIVIDUALS.keys())
def test_invalid_individual_fails_eval(gene_id):
    """eval.py should fail on known-bad individuals with the expected error."""
    model_name = f'model_{gene_id}'
    expected_error = INVALID_INDIVIDUALS[gene_id]

    result = subprocess.run(
        ['uv', 'run', 'python', EVAL_SCRIPT,
         '--model', model_name,
         '--variant_dir', INDIVIDUALS_DIR],
        cwd=ROOT, capture_output=True, text=True, timeout=10000
    )

    assert result.returncode != 0, \
        f"{gene_id}: expected failure but eval.py succeeded"
    assert expected_error in result.stderr, \
        f"{gene_id}: expected '{expected_error}' in stderr, got:\n{result.stderr[-500:]}"


# ── 4. Mutation produces valid Python (needs LLM server) ──────────────────

@pytest.mark.parametrize('gene_id', VALID_MUTATE)
def test_mutation_produces_valid_model(gene_id, tmp_path):
    """Mutate a frozen individual via llm_mutation.py, check output is valid."""
    prompt = os.path.join(ROOT, 'tests', 'operators', 'test_prompt_1.txt')
    if not os.path.isfile(prompt):
        pytest.skip("test_prompt_1.txt not found")

    parent = os.path.join(INDIVIDUALS_DIR, f'model_{gene_id}.py')
    child = str(tmp_path / f'mutated_{gene_id}.py')

    result = subprocess.run(
        ['uv', 'run', 'python', 'src/llm_mutation.py',
         parent, child, prompt,
         '--top_p', '0.1', '--temperature', '0.17',
         '--apply_quality_control', 'False'],
        cwd=ROOT, capture_output=True, text=True, timeout=10000
    )

    assert result.returncode == 0, \
        f"{gene_id}: llm_mutation.py failed: {result.stderr[-300:]}"
    assert os.path.isfile(child), \
        f"{gene_id}: output file not created"

    with open(child) as f:
        code = f.read()
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        pytest.xfail(f"{gene_id}: LLM produced invalid Python (non-deterministic): {e}")

    classes = [n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == 'Model']
    if len(classes) < 1:
        pytest.xfail(f"{gene_id}: mutated model has no Model class (non-deterministic)")
    methods = {n.name for n in ast.walk(classes[0])
               if isinstance(n, ast.FunctionDef)}
    if 'fit' not in methods:
        pytest.xfail(f"{gene_id}: mutated Model missing fit() (non-deterministic)")
    if 'predict' not in methods:
        pytest.xfail(f"{gene_id}: mutated Model missing predict() (non-deterministic)")


# ── 5. Crossover produces valid Python (needs LLM server) ─────────────────

# Verified crossover pairs from partner testing
CROSSOVER_PAIRS = [
    ('xXx0MtgG8LTm4mSVWscg4s97g0W', 'xXx09ejihbvx7yUX9JYxtIVPGDs'),
    ('xXx0MtgG8LTm4mSVWscg4s97g0W', 'xXx0UVvEW16rb3wrfuDSAemYblI'),
    ('xXx0LFCBTNE4Wx7KUjfsG1nZc4I', 'xXx07Gk5iNRGngQLpVQwbMDAAS1'),
    ('xXx0LFCBTNE4Wx7KUjfsG1nZc4I', 'xXx09ejihbvx7yUX9JYxtIVPGDs'),
    ('xXx0LFCBTNE4Wx7KUjfsG1nZc4I', 'xXx04jaaxCjDADdZfQejz3gbNFD'),
]


@pytest.mark.parametrize('gene_id_x, gene_id_y', CROSSOVER_PAIRS)
def test_crossover_function(gene_id_x, gene_id_y):
    """Cross two frozen individuals via llm_crossover.py, check output is valid."""
    input_filename_x = os.path.join(INDIVIDUALS_DIR, f'model_{gene_id_x}.py')
    input_filename_y = os.path.join(INDIVIDUALS_DIR, f'model_{gene_id_y}.py')

    crossed_output = os.path.join(RESULTS_DIR, f'model_crossed_{gene_id_x}_{gene_id_y}.py')

    result = subprocess.run(
        ['uv', 'run', 'python', CROSSOVER_SCRIPT, input_filename_x, input_filename_y, crossed_output],
        cwd=ROOT, capture_output=True, text=True, timeout=10000
    )

    # Pipeline must work
    assert result.returncode == 0, \
        f"({gene_id_x}, {gene_id_y}): crossover script failed: {result.stderr[-300:]}"
    assert 'job done' in result.stdout.lower(), \
        f"({gene_id_x}, {gene_id_y}): 'job done' not in output"
    assert os.path.isfile(crossed_output), \
        f"({gene_id_x}, {gene_id_y}): crossed output file not created at {crossed_output}"

    # LLM output quality — xfail if the LLM produces invalid code
    with open(crossed_output) as f:
        code = f.read()
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        pytest.xfail(f"({gene_id_x}, {gene_id_y}): LLM produced invalid Python (non-deterministic): {e}")

    classes = [n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == 'Model']
    if len(classes) < 1:
        pytest.xfail(f"({gene_id_x}, {gene_id_y}): LLM output missing Model class (non-deterministic)")

    methods = {n.name for n in ast.walk(classes[0])
               if isinstance(n, ast.FunctionDef)}
    if 'fit' not in methods:
        pytest.xfail(f"({gene_id_x}, {gene_id_y}): LLM output missing fit() (non-deterministic)")
    if 'predict' not in methods:
        pytest.xfail(f"({gene_id_x}, {gene_id_y}): LLM output missing predict() (non-deterministic)")



# ── 6. LLM creates a valid individual from seed (needs LLM server) ────────

def test_create_individual(tmp_path):
    """LLM mutation produces a valid model file from the seed."""
    prompt = os.path.join(ROOT, 'tests', 'operators', 'test_prompt_1.txt')
    output = str(tmp_path / "model_test.py")

    result = subprocess.run(
        ['uv', 'run', 'python', 'src/llm_mutation.py',
         f'{SOTA}/model.py', output, prompt,
         '--top_p', '0.1', '--temperature', '0.17',
         '--apply_quality_control', 'False'],
        cwd=ROOT, capture_output=True, text=True, timeout=600
    )
    assert result.returncode == 0, f"LLM mutation failed: {result.stderr[-300:]}"
    assert os.path.isfile(output), "Model file not created"
    ast.parse(open(output).read())
