import sys
# We are two directories down from run_improved, but running from its location
sys.path.append('./')
print(sys.path)
import run_improved
import os

def test_individual():
    individual = run_improved.toolbox.individual(llm_model=run_improved.DEFAULT_LLM_MODEL)
    assert os.path.exists(os.path.join(run_improved.OUTPUT_DIR, str(run_improved.GENERATION), f'{individual[0]}.sh'))
