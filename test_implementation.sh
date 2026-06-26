#!/bin/bash
echo "=== Verification of LLM_AVAIL Implementation ==="
echo ""
echo "1. Checking constants..."
grep -q "LLM_AVAIL = True" src/cfg/constants_titanic.py && echo "✓ LLM_AVAIL constant found" || echo "✗ LLM_AVAIL not found"
grep -q "SEED_MODELS_DIR" src/cfg/constants_titanic.py && echo "✓ SEED_MODELS_DIR constant found" || echo "✗ SEED_MODELS_DIR not found"

echo ""
echo "2. Checking llm_utils..."
grep -q "def select_random_seed_model" src/llm_utils.py && echo "✓ select_random_seed_model function found" || echo "✗ Function not found"

echo ""
echo "3. Checking run_improved.py imports..."
grep -q "select_random_seed_model" run_improved.py && echo "✓ Import added" || echo "✗ Import missing"

echo ""
echo "4. Checking run_improved.py modifications..."
grep -q "if not LLM_AVAIL:" run_improved.py && echo "✓ LLM_AVAIL checks added" || echo "✗ Checks missing"
grep -q "RANDOM_CREATED" run_improved.py && echo "✓ Random ancestry tracking added" || echo "✗ Ancestry tracking missing"
grep -q "--llm_avail" run_improved.py && echo "✓ CLI argument added" || echo "✗ CLI argument missing"

echo ""
echo "5. Checking seed models directory..."
[ -d "sota/Titanic/models/llmge_models_seed" ] && echo "✓ Seed directory exists" || echo "✗ Seed directory missing"
[ -f "sota/Titanic/models/llmge_models_seed/model_seed001.py" ] && echo "✓ Base seed model exists" || echo "✗ Base seed missing"

echo ""
echo "6. Counting modifications in run_improved.py..."
echo "   - LLM_AVAIL checks: $(grep -c 'if not LLM_AVAIL:' run_improved.py)"
echo "   - Random operations: $(grep -c 'RANDOM_' run_improved.py)"
echo ""
echo "=== Verification Complete ==="
