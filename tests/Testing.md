# Testing Documentation

## Overview
We have created unit tests dedicated to core functionalities of LLMGE. The unit tests help to validate critical stages of the evolutionary pipeline
such as the individual evaluation and the LLM-guided crossover. These tests help to ensure that the models being generate are valid Python
containing a valid class and methods. And are producing FP/FN results during the evalution on the Titantic dataset. These tests are built using pytest and some tests will require you to be on the Pace Ice cluster with an active LLM server.

## The Unit Tests
### test_individuals.py
This unit test focuses on the validity and evaluation on the individuals. Each individual will generate a python file, which contains the 'Model' with the required methods. The unit test will verify that each individual parses as valid python, containing the expected class structure, and produces bounded FP/FN results when evaluated by 'eval.py'.

### test_crossover.py
Tests the LLM-guided crossover function on pairs of frozen individuals. Given gene x and gene y, 'llm_crossover.py' submits their differing code segments to the LLM server to generate a new crossed individual. The unit test checks that the crossover script runs successfully, produces an output file, and that the resulting crossed individual is valid Python containing a `Model` class with the right methods. Pairs are randomly sampled from all possible combinations of the frozen individuals.

## Prerequistes
What is required before running:
- The LLM server must be running (sbatch server.sh) for unit tests such as crossover and mutation.

## Running the Tests
uv run pytest tests/ -v